# Memory Optimization Example Programs

This document describes the example programs created to demonstrate and test memory optimization features in the PTX Virtual Machine.

## Example Program Overview

The following example programs have been created to test different aspects of the memory optimization system:

| Example | Description | Focus Area | Optimization Type |
|--------|-------------|------------|-------------------|
| coalesced_memory_example.ptx | Demonstrates coalesced memory access patterns | Global memory | Coalescing |
| strided_memory_example.ptx | Demonstrates strided memory access patterns | Global memory | Coalescing |
| shared_memory_example.ptx | Demonstrates shared memory usage with bank conflicts | Shared memory | Bank conflict detection |
| tlb_memory_example.ptx | Demonstrates virtual memory access patterns | Virtual memory | TLB handling |
| complex_memory_example.ptx | Comprehensive test of multiple memory optimizations | All areas | Multiple optimizations |

## Coalesced Memory Example

This example demonstrates coalesced memory access patterns where threads in a warp access consecutive memory locations.

### Purpose
- Test memory coalescing optimization
- Measure cache hit rate improvements
- Demonstrate bandwidth utilization

### Code
```ptx
.version 7.0
.target sm_50
.address_size 64

.entry coalesced_kernel (
    .param .u64 data
)
{
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    
    // Get thread ID
    .reg .s32 %tid;
    mov.u32 %tid, %tid.x;
    
    // Calculate address - consecutive access
    mul.wide.s32 %rd1, %tid, 4;
    add.u64 %rd2, %data, %rd1;
    
    // Load data
    ld.global.f32 %f1, [%rd2];
    
    // Store data back
    add.f32 %f2, %f1, 1.0;
    st.global.f32 [%rd2], %f2;
    
    ret;
}
```

### Expected Optimization
- High cache hit rate (>90%)
- High memory bandwidth (>200 GB/s)
- No bank conflicts (for global memory)

## Strided Memory Example

This example demonstrates strided memory access patterns where threads access memory with a regular pattern.

### Purpose
- Test memory coalescing optimization for strided access
- Measure cache performance
- Demonstrate TLB efficiency

### Code
```ptx
.version 7.0
.target sm_50
.address_size 64

.entry strided_kernel (
    .param .u64 data
)
{
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    
    // Get thread ID
    .reg .s32 %tid;
    mov.u32 %tid, %tid.x;
    
    // Calculate address - strided access
    mul.wide.s32 %rd1, %tid, 16;  // 16-byte stride
    add.u64 %rd2, %data, %rd1;
    
    // Load data
    ld.global.f32 %f1, [%rd2];
    
    // Store data back
    add.f32 %f2, %f1, 1.0;
    st.global.f32 [%rd2], %f2;
    
    ret;
}
```

### Expected Optimization
- Moderate cache hit rate (80-85%)
- Moderate memory bandwidth (160-180 GB/s)
- TLB hit rate >95%

## Shared Memory Example

This example demonstrates shared memory usage and tests bank conflict detection.

### Purpose
- Test shared memory bank conflict detection
- Demonstrate warp divergence handling
- Measure shared memory performance

### Code
```ptx
.version 7.0
.target sm_50
.address_size 64

.entry shared_memory_kernel (
    .param .u64 data
)
{
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    
    // Shared memory allocation
    .shared .align 4 .u8 smem[128];
    
    // Get thread ID
    .reg .s32 %tid;
    mov.u32 %tid, %tid.x;
    
    // Calculate address - potential bank conflict
    mul.wide.s32 %rd1, %tid, 4;  // 4-byte stride
    
    // Load data from shared memory
    ld.shared.f32 %f1, [smem + %rd1];
    
    // Store data back to shared memory
    add.f32 %f2, %f1, 1.0;
    st.shared.f32 [smem + %rd1], %f2;
    
    ret;
}
```

### Expected Optimization
- Zero bank conflicts (with optimization)
- Shared memory bandwidth >180 GB/s
- Bank conflict rate 0%

## TLB Memory Example

This example demonstrates virtual memory access patterns and TLB efficiency.

### Purpose
- Test TLB and page fault handling
- Measure virtual memory performance
- Demonstrate page fault reduction

### Code
```ptx
.version 7.0
.target sm_50
.address_size 64

.entry tlb_kernel (
    .param .u64 data
)
{
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    
    // Get thread ID
    .reg .s32 %tid;
    mov.u32 %tid, %tid.x;
    
    // Calculate address - random access
    mul.wide.s32 %rd1, %tid, 4096;  // Page-sized stride
    add.u64 %rd2, %data, %rd1;
    
    // Load data
    ld.global.f32 %f1, [%rd2];
    
    // Store data back
    add.f32 %f2, %f1, 1.0;
    st.global.f32 [%rd2], %f2;
    
    ret;
}
```

### Expected Optimization
- TLB hit rate >90%
- Page faults = 0 after first run
- Memory bandwidth >140 GB/s

## Complex Memory Example

This comprehensive example tests multiple memory optimizations simultaneously.

### Purpose
- Test overall memory optimization effectiveness
- Measure performance with mixed access patterns
- Demonstrate multiple optimizations working together

### Code
```ptx
.version 7.0
.target sm_50
.address_size 64

.entry complex_kernel (
    .param .u64 data
)
{
    .reg .f32 %f<8>;
    .reg .u64 %rd<8>;
    .reg .s32 %tid;
    
    // Get thread ID
    mov.u32 %tid, %tid.x;
    
    // Shared memory allocation
    .shared .align 4 .u8 smem[1024];
    
    // Coalesced global memory access
    mul.wide.s32 %rd1, %tid, 4;
    add.u64 %rd2, %data, %rd1;
    ld.global.f32 %f1, [%rd2];
    add.f32 %f2, %f1, 1.0;
    st.global.f32 [%rd2], %f2;
    
    // Shared memory access with potential bank conflicts
    mul.wide.s32 %rd3, %tid, 4;
    ld.shared.f32 %f3, [smem + %rd3];
    add.f32 %f4, %f3, 2.0;
    st.shared.f32 [smem + %rd3], %f4;
    
    // Strided global memory access
    mul.wide.s32 %rd4, %tid, 16;
    add.u64 %rd5, %data, %rd4;
    ld.global.f32 %f5, [%rd5];
    add.f32 %f6, %f5, 3.0;
    st.global.f32 [%rd5], %f6;
    
    // Virtual memory access
    mul.wide.s32 %rd6, %tid, 4096;
    add.u64 %rd7, %data, %rd6;
    ld.global.f32 %f7, [%rd7];
    add.f32 %f8, %f7, 4.0;
    st.global.f32 [%rd7], %f8;
    
    ret;
}
```

### Expected Optimization
- Cache hit rate >85%
- TLB hit rate >90%
- Bank conflicts = 0
- Memory bandwidth >200 GB/s
- IPC >1.5

## Build Instructions

The examples can be assembled using NVIDIA's PTXAS assembler or any compatible CUDA toolchain:
```bash
# Assemble the examples
ptxas -v -o coalesced_memory_example.ptx coalesced_memory_example.ptx
ptxas -v -o strided_memory_example.ptx strided_memory_example.ptx
ptxas -v -o shared_memory_example.ptx shared_memory_example.ptx
ptxas -v -o tlb_memory_example.ptx tlb_memory_example.ptx
ptxas -v -o complex_memory_example.ptx complex_memory_example.ptx
```

For testing with the VM:
```bash
# Run coalesced memory test
./ptx_vm examples/coalesced_memory_example.ptx

# Run shared memory test
./ptx_vm examples/shared_memory_example.ptx

# Run performance test with all optimizations
./ptx_vm examples/complex_memory_example.ptx
```

## Performance Test Instructions

To run the full performance test suite:
```bash
# Build the tests
mkdir build && cd build
cmake ..
make

# Run the memory performance tests
ctest -R "MemoryPerformanceTest"

# Run all performance tests
make test
```

The tests will output detailed performance metrics including:
- Instructions per cycle (IPC)
- Cache hit rate
- TLB hit rate
- Memory bandwidth
- Execution time
- Bank conflicts

## Expected Performance Results

### Coalesced Access
| Metric | Expected Value |
|--------|----------------|
| IPC | >1.5 |
| Cache Hit Rate | >90% |
| Memory Bandwidth | >200 GB/s |
| Execution Time | <500 ms |

### Strided Access
| Metric | Expected Value |
|--------|----------------|
| IPC | >1.2 |
| Cache Hit Rate | >80% |
| TLB Hit Rate | >95% |
| Memory Bandwidth | >160 GB/s |
| Execution Time | <600 ms |

### Shared Memory
| Metric | Expected Value |
|--------|----------------|
| IPC | >1.3 |
| Bank Conflicts | 0 |
| Shared Memory Bandwidth | >180 GB/s |
| Execution Time | <550 ms |

### TLB Handling
| Metric | Expected Value |
|--------|----------------|
| IPC | >1.1 |
| TLB Hit Rate | >90% |
| Page Faults | 0 |
| Memory Bandwidth | >140 GB/s |
| Execution Time | <700 ms |

### Complex Pattern
| Metric | Expected Value |
|--------|----------------|
| IPC | >1.8 |
| Cache Hit Rate | >90% |
| TLB Hit Rate | >95% |
| Shared Memory Bandwidth | >180 GB/s |
| Global Memory Bandwidth | >220 GB/s |
| Execution Time | <800 ms |

## Example Program Usage

To run any example:
```bash
./ptx_vm examples/<example_name>.ptx
```

To get detailed statistics:
```bash
./ptx_vm
> load examples/<example_name>.ptx
> profile performance.csv
> run
> dump
> exit
```

The performance data will be written to `performance.csv` for analysis.

## Example Program Development

When creating new example programs to test memory optimizations:

1. **Identify the optimization to test**
   - Coalescing
   - Bank conflicts
   - TLB efficiency
   - Mixed patterns

2. **Design the memory access pattern**
   - Coalesced: Consecutive memory accesses
   - Strided: Regular spaced accesses
   - Gather/Scatter: Random accesses
   - Shared memory: Accesses that could cause bank conflicts

3. **Implement the kernel**
   - Keep it simple and focused
   - Use standard memory access patterns
   - Include both read and write operations
   - Use standard PTX syntax

4. **Test and validate**
   - Run with different optimization settings
   - Verify correctness of results
   - Measure performance improvements
   - Compare with baseline execution

5. **Document the example**
   - Describe the access pattern
   - Specify expected optimizations
   - Document expected performance
   - Include execution instructions

## Integration with Build System

The example programs are integrated into the CMake build system:
```cmake
# examples/CMakeLists.txt
add_custom_target(memory_examples
    COMMAND ${CMAKE_COMMAND} -E echo "Memory optimization examples: coalesced, strided, shared, tlb, complex"
    COMMAND ${CMAKE_COMMAND} -E echo "Assemble with ptxas or use directly with VM"
)

# Add to main build
add_subdirectory(examples)
```

## Future Example Improvements
Planned enhancements for example programs include:
- Better documentation
- More complex access patterns
- Enhanced performance measurement
- Better error handling
- More comprehensive test coverage
- Integration with VM profiler
- Support for multiple data types
- Better validation of results
- Enhanced visualization of access patterns
- Integration with test framework
- Support for parameterized tests
- Better error checking
- Enhanced instruction mix
- Better memory footprint control
- Support for different optimization levels
- Better integration with performance counters
- Enhanced register usage
- Better control flow
- Enhanced memory access patterns
- Better support for debugging