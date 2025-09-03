#include <stdexcept>
#include <stack>
#include <vector>

class MemoryPoolAllocator
{
public:
  MemoryPoolAllocator(const size_t num_blocks, const size_t block_size_bytes)
    : num_blocks_(num_blocks),
      block_size_bytes_(block_size_bytes),
      buffer_(num_blocks * block_size_bytes),
      free_list_()
  {
    // Fill free list with all block indices
    for (size_t i = 0; i < num_blocks_; ++i)
      free_list_.push(i);
  }

  // Allocate one fixed-size block
  void* Allocate()
  {
    if (free_list_.empty())
      throw std::bad_alloc();

    size_t block_index = free_list_.top();
    free_list_.pop();
    return buffer_.data() + block_index * block_size_bytes_;
  }

  // Deallocate a block given its pointer
  void Deallocate(void* ptr)
  {
    auto base = buffer_.data();
    auto byte_ptr = static_cast<std::byte*>(ptr);

    if (byte_ptr < base || byte_ptr >= base + buffer_.size())
      throw std::runtime_error("Pointer does not belong to this pool");

    size_t offset = byte_ptr - base;
    if (offset % block_size_bytes_ != 0)
      throw std::runtime_error("Pointer not aligned to block boundary");

    size_t block_index = offset / block_size_bytes_;
    free_list_.push(block_index);
  }

  size_t Capacity() const { return num_blocks_; }
  size_t FreeBlocks() const { return free_list_.size(); }
  size_t GetBufferSize() const { return buffer_.size(); }

private:
  size_t num_blocks_;
  size_t block_size_bytes_;
  std::vector<std::byte> buffer_;
  std::stack<size_t> free_list_;
};
