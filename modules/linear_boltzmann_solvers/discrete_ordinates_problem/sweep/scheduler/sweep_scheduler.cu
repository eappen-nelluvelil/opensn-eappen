#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/sweep_scheduler.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbc_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbcd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/reflecting_boundary.h"
#include "cuda_runtime.h"

namespace opensn
{



void
SweepScheduler::ScheduleAlgoFIFOAsync(SweepChunk& sweep_chunk)
{
	CBCSweepChunk& cbc_sweep_chunk = dynamic_cast<CBCSweepChunk&>(sweep_chunk);

	cbc_sweep_chunk.CopyPhiAndSrcToDevice();

	std::vector<CBC_AngleSet*> angle_sets;
	for (auto& angle_set_group : angle_agg_.angle_set_groups)
	  for (auto& angle_set : angle_set_group.GetAngleSets())
	    angle_sets.push_back(dynamic_cast<CBC_AngleSet*>(angle_set.get()));

	const size_t num_angle_sets = angle_sets.size();

	// opensn::log.Log() << "Number of angle sets: " << num_angle_sets;

	for (auto* as : angle_sets)
	{
		auto& task_list = as->GetCurrentTaskList();
		if (task_list.empty())
			task_list = as->GetSPDS().GetTaskList();
	}

	std::vector<bool> executed(num_angle_sets, false);
	std::vector<bool> boundary_data_set(num_angle_sets, false);

	bool all_finished = false;
	while (not all_finished)
	{
		all_finished = true;

		// Check execution status, receive data, and send data
		for (size_t i = 0; i < num_angle_sets; ++i)
		{
			if (executed[i])
				continue;

			all_finished = false;

			auto* as = angle_sets[i];
			auto* comm = dynamic_cast<CBC_ASynchronousCommunicator*>(as->GetCommunicator());

			auto tasks_received = comm->ReceiveData();
			auto& task_list = as->GetCurrentTaskList();

			for (uint64_t t : tasks_received)
			  --task_list[t].num_dependencies;

			comm->SendData();
		}

		// Check boundary readiness
		for (size_t i = 0; i < num_angle_sets; ++i)
		{
			if (executed[i])
			  continue;

			auto* as = angle_sets[i];
			bool boundaries_ready = true;
			
			for (auto& [bid, boundary] : as->GetBoundaries())
			{
				if (not boundary->CheckAnglesReadyStatus(as->GetAngleIndices()))
				{
					boundaries_ready = false;
					break;
				}
			}

			if (boundaries_ready and not boundary_data_set[i])
			{
				reinterpret_cast<CBC_FLUDS&>(as->GetFLUDS()).GetAndSetBoundaryPsiData(sweep_chunk, dynamic_cast<AngleSet&>(*as));
				// reinterpret_cast<CBC_FLUDS&>(as->GetFLUDS()).GetAndSetBoundaryPsiDataAsync(sweep_chunk, dynamic_cast<AngleSet&>(*as));
				boundary_data_set[i] = true;
			}
		}

		// Collect ready tasks
		std::vector<std::vector<Task*>> ready_tasks(num_angle_sets);
		for (size_t i = 0; i < num_angle_sets; ++i)
		{
			if (executed[i])
				continue;

			auto& current_task_list = angle_sets[i]->GetCurrentTaskList();
			for (auto& task : current_task_list)
			{
				if (task.num_dependencies == 0 and not task.completed)
					ready_tasks[i].push_back(&task);
			}
		}

		// Launch concurrent kernels
		for (size_t i = 0; i < num_angle_sets; ++i)
		{
			if (executed[i] or ready_tasks[i].empty())
				continue;

			auto* as = angle_sets[i];
			auto& fluds = dynamic_cast<CBC_FLUDS&>(as->GetFLUDS());

			cbc_sweep_chunk.GPUSweepAsync(dynamic_cast<AngleSet&>(*as), ready_tasks[i]);
		}

		// Synchronize streams
		// for (size_t i = 0; i < num_angle_sets; ++i)
		// {
		// 	if (executed[i] or ready_tasks[i].empty())
		// 	  continue;

    // 	auto* as = angle_sets[i];
    // 	cudaStream_t stream = reinterpret_cast<cudaStream_t>(as->stream_ptr);
    // 	cudaStreamSynchronize(stream);
    // }
    cudaDeviceSynchronize();

		// Set outgoing non-local and reflecting boundary data
		for (size_t i = 0; i < num_angle_sets; ++i)
		{
			if (executed[i] or ready_tasks[i].empty())
			  continue;

			auto* as = angle_sets[i];

			auto& fluds = dynamic_cast<CBC_FLUDS&>(as->GetFLUDS());
			// fluds.SetNonlocalAndReflectingBoundaryPsiDataAsync(sweep_chunk, dynamic_cast<AngleSet&>(*as), ready_tasks[i]);
			fluds.SetNonlocalAndReflectingBoundaryPsiData(sweep_chunk, dynamic_cast<AngleSet&>(*as), ready_tasks[i]);

			auto& current_task_list = as->GetCurrentTaskList();
			auto* comm = dynamic_cast<CBC_ASynchronousCommunicator*>(as->GetCommunicator());

			for (auto* task : ready_tasks[i])
			{
				for (uint64_t succ : task->local_successors)
				{
					--current_task_list[succ].num_dependencies;
				}
				task->completed = true;
				comm->SendData();
			}
    }

    // Send data
    for (size_t i = 0; i < num_angle_sets; ++i)
		{
			if (executed[i])
				continue;

			auto* as = angle_sets[i];
			auto* comm = dynamic_cast<CBC_ASynchronousCommunicator*>(as->GetCommunicator());

			comm->SendData();
		}

		// Check angleset completion
		for (size_t i = 0; i < num_angle_sets; ++i)
		{
			if (executed[i])
			  continue;

			auto* as = angle_sets[i];
			auto& current_task_list = as->GetCurrentTaskList();
			auto* comm = dynamic_cast<CBC_ASynchronousCommunicator*>(as->GetCommunicator());

			bool done = std::all_of(current_task_list.begin(), current_task_list.end(),
									[](const Task& t) { return t.completed; });

			if (done && comm->SendData())
			{
				for (auto& [bid, boundary] : as->GetBoundaries())
				{
					boundary->UpdateAnglesReadyStatus(as->GetAngleIndices());
				}
				executed[i] = true;
			}
		}
	}

	cbc_sweep_chunk.CopyOutflowAndPhiFromDevice();

	// Receive delayed data
  opensn::mpi_comm.barrier();
  bool received_delayed_data = false;
  while (not received_delayed_data)
  {
    received_delayed_data = true;

    for (auto& angle_set_group : angle_agg_.angle_set_groups)
      for (auto& angle_set : angle_set_group.GetAngleSets())
      {
        if (angle_set->FlushSendBuffers() == AngleSetStatus::MESSAGES_PENDING)
          received_delayed_data = false;

        if (not angle_set->ReceiveDelayedData())
          received_delayed_data = false;
      }
  }

  // Reset all
  for (auto& angle_set_group : angle_agg_.angle_set_groups)
    for (auto& angle_set : angle_set_group.GetAngleSets())
      angle_set->ResetSweepBuffers();

  for (const auto& [bid, bndry] : angle_agg_.GetSimBoundaries())
  {
    if (bndry->GetType() == LBSBoundaryType::REFLECTING)
    {
      auto rbndry = std::static_pointer_cast<ReflectingBoundary>(bndry);
      rbndry->ResetAnglesReadyStatus();
    }
  }
}

}  // namespace opensn