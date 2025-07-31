# Divergence Handling Example Programs

This document describes the example programs created to demonstrate and test divergence handling in the PTX Virtual Machine.

## Example Program Overview

The following example programs have been created to test different aspects of the divergence handling system:

| Example | Description | Focus Area | Divergence Type |
|--------|-------------|------------|-----------------|
| simple_divergence.ptx | Demonstrates basic branch divergence | Control flow | Simple divergence |
| nested_divergence.ptx | Demonstrates nested branch divergence | Control flow | Nested divergence |
| complex_divergence.ptx | Demonstrates complex control flow with multiple divergence points | Control flow | Multiple divergence |
| long_divergence.ptx | Demonstrates long-running divergent code sections | Control flow | Prolonged divergence |
| reconvergence_divergence.ptx | Demonstrates proper reconvergence handling | Control flow | Divergence and reconvergence |

## Simple Divergence Example

This example demonstrates basic branch divergence where half the threads take one path and half take another.

### Purpose
- Test basic divergence handling
- Measure reconvergence accuracy
- Demonstrate simple branch divergence

### Code
```ptx
.version 7.0
.target sm_50
.address_size 64

.entry simple_divergence (
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
    
    // Simple branch divergence
    .reg .pred %p<2>;
    setp.lt.s32 %p1, %tid, 16;
    @%p1 bra.uni L1;
    
    // Path for threads >= 16
    add.f32 %f2, %f1, 1.0;
    st.global.f32 [%rd2], %f2;
    bra.uni L2;
    
L1:
    // Path for threads < 16
    add.f32 %f2, %f1, 2.0;
    st.global.f32 [%rd2], %f2;
    
L2:
    // Reconvergence point
    ret;
}
```

### Expected Behavior
- 50% of threads take each branch
- Proper reconvergence at L2
- Correct data written to memory

## Nested Divergence Example

This example demonstrates nested branch divergence where threads diverge multiple times.

### Purpose
- Test nested divergence handling
- Measure maximum divergence depth
- Demonstrate complex control flow

### Code
```ptx
.version 7.0
.target sm_50
.address_size 64

.entry nested_divergence (
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
    
    // First branch divergence
    .reg .pred %p<2>;
    setp.lt.s32 %p1, %tid, 16;
    @%p1 bra.uni L1;
    
    // Second level divergence for threads >= 16
    setp.lt.s32 %p1, %tid, 32;
    @%p1 bra.uni L2;
    
    // Path for threads 16-31
    add.f32 %f2, %f1, 1.0;
    st.global.f32 [%rd2], %f2;
    bra.uni L3;
    
L1:
    // Path for threads < 16
    add.f32 %f2, %f1, 2.0;
    st.global.f32 [%rd2], %f2;
    bra.uni L3;
    
L2:
    // Path for threads 16-31
    add.f32 %f2, %f1, 3.0;
    st.global.f32 [%rd2], %f2;
    
L3:
    // Reconvergence point
    ret;
}
```

### Expected Behavior
- First divergence at 16 threads
- Second divergence for threads 16-31
- Maximum divergence depth of 2
- Proper reconvergence at L3

## Complex Divergence Example

This example demonstrates complex control flow with multiple divergence points.

### Purpose
- Test complex divergence scenarios
- Measure divergence impact
- Demonstrate multiple divergence points

### Code
```ptx
.version 7.0
.target sm_50
.address_size 64

.entry complex_divergence (
    .param .u64 data
)
{
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    .reg .s32 %tid;
    
    // Get thread ID
    mov.u32 %tid, %tid.x;
    
    // Shared memory allocation
    .shared .align 4 .u8 smem[128];
    
    // Calculate address - consecutive access
    mul.wide.s32 %rd1, %tid, 4;
    add.u64 %rd2, %data, %rd1;
    
    // First divergence point
    .reg .pred %p<2>;
    setp.lt.s32 %p1, %tid, 8;
    @%p1 bra.uni L1;
    
    // Second divergence point
    setp.lt.s32 %p1, %tid, 16;
    @%p1 bra.uni L2;
    
    // Third divergence point
    setp.lt.s32 %p1, %tid, 24;
    @%p1 bra.uni L3;
    
    // Path for threads >= 24
    add.f32 %f1, %f1, 1.0;
    st.global.f32 [%rd2], %f1;
    bra.uni L4;
    
L1:
    // Path for threads < 8
    add.f32 %f1, %f1, 2.0;
    st.global.f32 [%rd2], %f1;
    bra.uni L4;
    
L2:
    // Path for threads 8-15
    add.f32 %f1, %f1, 3.0;
    st.global.f32 [%rd2], %f1;
    bra.uni L4;
    
L3:
    // Path for threads 16-23
    add.f32 %f1, %f1, 4.0;
    st.global.f32 [%rd2], %f1;
    
L4:
    // Reconvergence point
    ret;
}
```

### Expected Behavior
- Multiple divergence points
- Divergence depth of 1 for most threads
- Divergence depth of 2 for some threads
- Proper reconvergence at L4

## Long-Running Divergence Example

This example demonstrates long-running divergent code sections.

### Purpose
- Test prolonged divergence handling
- Demonstrate stack-based predication
- Measure divergence impact over time

### Code
```ptx
.version 7.0
.target sm_50
.address_size 64

.entry long_divergence (
    .param .u64 data
)
{
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    .reg .s32 %tid;
    .reg .s32 %i;
    
    // Get thread ID
    mov.u32 %tid, %tid.x;
    
    // Calculate address - consecutive access
    mul.wide.s32 %rd1, %tid, 4;
    add.u64 %rd2, %data, %rd1;
    
    // Load data
    ld.global.f32 %f1, [%rd2];
    
    // First divergence point
    .reg .pred %p<2>;
    setp.lt.s32 %p1, %tid, 16;
    @%p1 bra.uni L1;
    
    // Long-running code for threads >= 16
    mov.s32 %i, 0;
L2:
    add.f32 %f1, %f1, 1.0;
    add.s32 %i, %i, 1;
    setp.lt.s32 %p1, %i, 100;
    @%p1 bra.uni L2;
    
    st.global.f32 [%rd2], %f1;
    bra.uni L3;
    
L1:
    // Short path for threads < 16
    add.f32 %f1, %f1, 100.0;
    st.global.f32 [%rd2], %f1;
    
L3:
    // Reconvergence point
    ret;
}
```

### Expected Behavior
- Divergence with long-running code section
- Proper handling of extended divergence
- Correct reconvergence after long execution
- Stack-based predication for extended divergence

## Reconvergence Handling Example

This example demonstrates proper reconvergence handling with different control flow patterns.

### Purpose
- Test reconvergence point detection
- Demonstrate CFG-based reconvergence
- Measure reconvergence accuracy

### Code
```ptx
.version 7.0
.target sm_50
.address_size 64

.entry reconvergence_example (
    .param .u64 data
)
{
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    .reg .s32 %tid;
    
    // Get thread ID
    mov.u32 %tid, %tid.x;
    
    // Calculate address - consecutive access
    mul.wide.s32 %rd1, %tid, 4;
    add.u64 %rd2, %data, %rd1;
    
    // Load data
    ld.global.f32 %f1, [%rd2];
    
    // First divergence point
    .reg .pred %p<2>;
    setp.lt.s32 %p1, %tid, 16;
    @%p1 bra.uni L1;
    
    // Second divergence point
    setp.lt.s32 %p1, %tid, 32;
    @%p1 bra.uni L2;
    
    // Path for threads >= 32
    add.f32 %f2, %f1, 1.0;
    st.global.f32 [%rd2], %f2;
    bra.uni L3;
    
L1:
    // Path for threads < 16
    add.f32 %f2, %f1, 2.0;
    st.global.f32 [%rd2], %f2;
    bra.uni L3;
    
L2:
    // Path for threads 16-31
    add.f32 %f2, %f1, 3.0;
    st.global.f32 [%rd2], %f2;
    
L3:
    // Reconvergence point
    // Additional divergence point
    setp.lt.s32 %p1, %tid, 8;
    @%p1 bra.uni L4;
    
    // Path for threads >= 8
    add.f32 %f1, %f2, 4.0;
    st.global.f32 [%rd2], %f1;
    bra.uni L5;
    
L4:
    // Path for threads < 8
    add.f32 %f1, %f2, 5.0;
    st.global.f32 [%rd2], %f1;
    
L5:
    // Final reconvergence point
    ret;
}
```

### Expected Behavior
- Multiple reconvergence points
- Proper handling of nested divergence
- Correct CFG-based reconvergence
- Stack-based predication for complex flow

## Build Instructions

The examples can be assembled using NVIDIA's PTXAS assembler or any compatible CUDA toolchain:
```bash
# Assemble the examples
ptxas -v -o simple_divergence.ptx simple_divergence.ptx
ptxas -v -o nested_divergence.ptx nested_divergence.ptx
ptxas -v -o complex_divergence.ptx complex_divergence.ptx
ptxas -v -o long_divergence.ptx long_divergence.ptx
ptxas -v -o reconvergence_divergence.ptx reconvergence_divergence.ptx
```

For testing with the VM:
```bash
# Run simple divergence test
./ptx_vm examples/simple_divergence.ptx

# Run nested divergence test
./ptx_vm examples/nested_divergence.ptx

# Run performance test with all examples
./ptx_vm examples/complex_divergence.ptx
```

## Performance Test Instructions

To run the full performance test suite:
```bash
# Build the tests
mkdir build && cd build
cmake ..
make

# Run the divergence performance tests
ctest -R "DivergencePerformanceTest"

# Run all performance tests
make test
```

The tests will output detailed performance metrics including:
- Divergence events
- Reconvergence events
- Divergence depth
- Divergence impact
- Execution time

## Expected Performance Results

### Simple Divergence
| Metric | Expected Value |
|--------|----------------|
| Divergence Events | 1 |
| Reconvergence Events | 1 |
| Max Divergence Depth | 1 |
| Divergence Impact | < 1.0 |
| Execution Time | < 1000 ms |

### Nested Divergence
| Metric | Expected Value |
|--------|----------------|
| Divergence Events | 2 |
| Reconvergence Events | 2 |
| Max Divergence Depth | 2 |
| Divergence Impact | < 1.5 |
| Execution Time | < 1200 ms |

### Complex Divergence
| Metric | Expected Value |
|--------|----------------|
| Divergence Events | 3 |
| Reconvergence Events | 3 |
| Max Divergence Depth | 1 |
| Divergence Impact | < 1.0 |
| Execution Time | < 1100 ms |

### Long-Running Divergence
| Metric | Expected Value |
|--------|----------------|
| Divergence Events | 1 |
| Reconvergence Events | 1 |
| Max Divergence Depth | 1 |
| Divergence Impact | < 2.0 |
| Execution Time | < 1500 ms |

### Reconvergence Handling
| Metric | Expected Value |
|--------|----------------|
| Divergence Events | 2 |
| Reconvergence Events | 2 |
| Max Divergence Depth | 1 |
| Divergence Impact | < 1.0 |
| Execution Time | < 1300 ms |

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

When creating new example programs to test divergence handling:

1. **Identify the divergence pattern to test**
   - Simple divergence
   - Nested divergence
   - Multiple divergence points
   - Long-running divergence
   - Complex control flow

2. **Design the control flow pattern**
   - Simple branch
   - Nested branches
   - Multiple independent branches
   - Complex CFG with reconvergence points
   - Long-running code sections

3. **Implement the kernel**
   - Keep it simple and focused
   - Use standard PTX control flow
   - Include both divergent and non-divergent paths
   - Use standard PTX syntax
   - Include validation of results

4. **Test and validate**
   - Run with different algorithms
   - Verify correctness of results
   - Measure performance impact
   - Compare with baseline execution

5. **Document the example**
   - Describe the divergence pattern
   - Specify expected divergence behavior
   - Document expected performance metrics
   - Include execution instructions

## Integration with Build System

The example programs are integrated into the CMake build system:
```cmake
# examples/CMakeLists.txt
add_custom_target(divergence_examples
    COMMAND ${CMAKE_COMMAND} -E echo "Divergence handling examples: simple, nested, complex, long, reconvergence"
    COMMAND ${CMAKE_COMMAND} -E echo "Assemble with ptxas or use directly with VM"
)

# Add to main build
add_subdirectory(examples)
```

## Future Example Improvements
Planned enhancements for example programs include:
- Better documentation
- More complex divergence patterns
- Enhanced performance measurement
- Better error handling
- More comprehensive test coverage
- Integration with VM profiler
- Support for parameterized tests
- Better validation of results
- Enhanced visualization of divergence patterns
- Integration with test framework
- Support for different optimization levels
- Better integration with performance counters
- Enhanced register usage
- Better control flow
- Enhanced divergence patterns
- Support for different divergence rates
- Enhanced instruction mix
- Better memory access patterns
- Enhanced divergence tracking
- Better divergence impact measurement
- Enhanced divergence handling validation
- Better integration with debugging tools