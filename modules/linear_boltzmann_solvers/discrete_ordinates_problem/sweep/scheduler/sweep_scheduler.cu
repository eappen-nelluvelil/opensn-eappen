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


/*
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

  std::vector<bool> executed(num_angle_sets, false);
  std::vector<bool> boundary_data_set(num_angle_sets, false);
  std::vector<std::vector<Task*>> ready_tasks(num_angle_sets);

  // Get current task list
  for (auto* as : angle_sets)
  {
    auto& current_task_list = as->GetCurrentTaskList();
    if (current_task_list.empty())
      current_task_list = as->GetSPDS().GetTaskList();
  }

  size_t executed_anglesets = 0;
  while (executed_anglesets < num_angle_sets)
  {
    // Check execution status, receive data, and send data
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i])
        continue;

      auto* as = angle_sets[i];
      auto* comm = dynamic_cast<CBC_ASynchronousCommunicator*>(as->GetCommunicator());
      auto& current_task_list = as->GetCurrentTaskList();

      auto tasks_that_received_data = comm->ReceiveData();

      for (uint64_t t : tasks_that_received_data)
        --current_task_list[t].num_dependencies;

      comm->SendData();
    }

    // Check boundary readiness
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i])
        continue;

      auto* as = angle_sets[i];

      if (not boundary_data_set[i])
      {
        bool boundaries_ready = true;

        for (auto& [bid, boundary] : as->GetBoundaries())
        {
          if (not boundary->CheckAnglesReadyStatus(as->GetAngleIndices()))
          {
            boundaries_ready = false;
            break;
          }
        }

        if (boundaries_ready)
        {
          reinterpret_cast<CBC_FLUDS&>(as->GetFLUDS()).GetAndSetBoundaryPsiDataAsync(sweep_chunk,
dynamic_cast<AngleSet&>(*as)); boundary_data_set[i] = true;
        }
      }

      if (boundary_data_set[i])
      {
        ready_tasks[i].clear();
        for (auto& task : as->GetCurrentTaskList())
          if (task.num_dependencies == 0 and not task.completed)
            ready_tasks[i].push_back(&task);
      }
    }

    // Launch concurrent kernels
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i] or not boundary_data_set[i] or ready_tasks[i].empty())
        continue;
      cbc_sweep_chunk.GPUSweepAsync(*angle_sets[i], ready_tasks[i]);
    }

    // Set outgoing non-local and reflecting boundary data
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i] or not boundary_data_set[i] or ready_tasks[i].empty())
        continue;

      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(angle_sets[i]->stream_ptr));

      auto& fluds = dynamic_cast<CBC_FLUDS&>(angle_sets[i]->GetFLUDS());
      fluds.SetNonlocalAndReflectingBoundaryPsiDataAsync(sweep_chunk, *angle_sets[i],
ready_tasks[i]);

      auto& current_task_list = angle_sets[i]->GetCurrentTaskList();
      auto* comm = dynamic_cast<CBC_ASynchronousCommunicator*>(angle_sets[i]->GetCommunicator());

      for (auto* task : ready_tasks[i])
      {
        for (uint64_t succ : task->local_successors)
          --current_task_list[succ].num_dependencies;
        task->completed = true;
        // comm->SendData();
      }

      comm->SendData();
    }

    // Check angleset completion
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i])
        continue;

      auto& current_task_list = angle_sets[i]->GetCurrentTaskList();
      auto* comm = dynamic_cast<CBC_ASynchronousCommunicator*>(angle_sets[i]->GetCommunicator());

      bool done = std::all_of(current_task_list.begin(), current_task_list.end(),
                  [](const Task& t) { return t.completed; });

      if (done && comm->SendData())
      {
        for (auto& [bid, boundary] : angle_sets[i]->GetBoundaries())
          boundary->UpdateAnglesReadyStatus(angle_sets[i]->GetAngleIndices());
        executed[i] = true;
        ++executed_anglesets;
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
*/

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

  std::vector<bool> executed(num_angle_sets, false);
  std::vector<bool> boundary_data_set(num_angle_sets, false);
  std::vector<bool> kernel_in_flight(num_angle_sets, false);
  std::vector<std::vector<Task*>> ready_tasks(num_angle_sets);
  std::vector<std::vector<Task*>> in_flight_tasks(num_angle_sets);

  for (auto* as : angle_sets)
  {
    auto& current_task_list = as->GetCurrentTaskList();
    if (current_task_list.empty())
      current_task_list = as->GetSPDS().GetTaskList();
  }

  size_t executed_anglesets = 0;
  while (executed_anglesets < num_angle_sets)
  {
    // Poll for completed kernels and process D2H + MPI
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (not kernel_in_flight[i])
        continue;

      cudaStream_t stream = reinterpret_cast<cudaStream_t>(angle_sets[i]->stream_ptr);
      cudaError_t status = cudaStreamQuery(stream);

      if (status == cudaSuccess)
      {
        auto& fluds = dynamic_cast<CBC_FLUDS&>(angle_sets[i]->GetFLUDS());
        fluds.SetNonlocalAndReflectingBoundaryPsiDataAsync(
          sweep_chunk, *angle_sets[i], in_flight_tasks[i]);

        auto& current_task_list = angle_sets[i]->GetCurrentTaskList();
        for (auto* task : in_flight_tasks[i])
        {
          for (uint64_t succ : task->local_successors)
            --current_task_list[succ].num_dependencies;
          task->completed = true;
        }

        auto* comm =
          dynamic_cast<CBC_ASynchronousCommunicator*>(angle_sets[i]->GetCommunicator());
        comm->SendData();

        in_flight_tasks[i].clear();
        kernel_in_flight[i] = false;
      }
    }

    // Receive MPI data and update dependencies
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i])
        continue;

      auto* comm =
        dynamic_cast<CBC_ASynchronousCommunicator*>(angle_sets[i]->GetCommunicator());
      auto& current_task_list = angle_sets[i]->GetCurrentTaskList();

      for (uint64_t t : comm->ReceiveData())
        --current_task_list[t].num_dependencies;

      comm->SendData();
    }

    // Set boundary data and collect ready tasks
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i] or kernel_in_flight[i])
        continue;

      auto* as = angle_sets[i];

      if (not boundary_data_set[i])
      {
        bool boundaries_ready = true;
        for (auto& [bid, boundary] : as->GetBoundaries())
        {
          if (not boundary->CheckAnglesReadyStatus(as->GetAngleIndices()))
          {
            boundaries_ready = false;
            break;
          }
        }

        if (boundaries_ready)
        {
          dynamic_cast<CBC_FLUDS&>(as->GetFLUDS())
            .GetAndSetBoundaryPsiDataAsync(sweep_chunk, *as);
          boundary_data_set[i] = true;
        }
      }

      if (boundary_data_set[i])
      {
        ready_tasks[i].clear();
        for (auto& task : as->GetCurrentTaskList())
          if (task.num_dependencies == 0 and not task.completed)
            ready_tasks[i].push_back(&task);
      }
    }

    // Launch kernels
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i] or not boundary_data_set[i] or ready_tasks[i].empty() or kernel_in_flight[i])
        continue;

      cbc_sweep_chunk.GPUSweepAsync(*angle_sets[i], ready_tasks[i]);

      in_flight_tasks[i] = std::move(ready_tasks[i]);
      kernel_in_flight[i] = true;
    }

    // Check angleset completion
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i] or kernel_in_flight[i])
        continue;

      auto& current_task_list = angle_sets[i]->GetCurrentTaskList();
      auto* comm =
        dynamic_cast<CBC_ASynchronousCommunicator*>(angle_sets[i]->GetCommunicator());

      bool all_done = std::all_of(current_task_list.begin(),
                                  current_task_list.end(),
                                  [](const Task& t) { return t.completed; });

      if (all_done and comm->SendData())
      {
        for (auto& [bid, boundary] : angle_sets[i]->GetBoundaries())
          boundary->UpdateAnglesReadyStatus(angle_sets[i]->GetAngleIndices());
        executed[i] = true;
        ++executed_anglesets;
      }
    }
  }

  cbc_sweep_chunk.CopyOutflowAndPhiFromDevice();

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