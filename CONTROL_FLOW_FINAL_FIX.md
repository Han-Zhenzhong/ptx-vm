# Control Flow Bug Fixes - Final Summary

## Test Result Evolution

### Initial Problem
- "Invalid SETP instruction format" errors
- "Invalid BRA instruction format" errors  
- Branch counter = 0 (no branches)
- Result = 40 (4 iterations instead of 5)

### After Parser Fixes
- SETP errors: ✅ FIXED
- BRA errors: ✅ FIXED
- Branch counter = 0: ✅ FIXED (branches now count)
- Result = 80 (8 iterations instead of 5) ⚠️ NEW ISSUE

### Final Issue: Multi-Warp Execution
**Result was 80 instead of 50 because:**
- Executor initialized with 4 warps
- Each instruction executed 4 times (once per warp)
- All warps shared the same register file (WRONG!)
- Loop counter incremented: 0→4→8 (4 times per iteration)
- Each warp ran 8 iterations: 8 × 10 = 80

## All Fixes Applied

### 1. Parser: Branch Instructions Have No Destination (CRITICAL)
**File:** `src/parser/parser.cpp` line 336

**Problem:** Parser assumed ALL instructions have format: `opcode dest, src1, src2`  
For `bra loop_start`, it set dest=`loop_start` and sources=`[]` (empty)

**Fix:**
```cpp
if (instr.opcode == "bra" || instr.opcode == "call" || 
    instr.opcode == "ret" || instr.opcode == "exit")
{
    // All operands are sources for branch/call/return
    for (const auto& op : operands) {
        instr.sources.push_back(op);
    }
}
```

### 2. Parser: Recognize Predicate Registers (`%p`)
**File:** `src/parser/parser.cpp` line 833

**Problem:** `%p1` was parsed as REGISTER type instead of PREDICATE type

**Fix:** Check for `%p` pattern before generic `%` pattern:
```cpp
if (s.size() >= 3 && s[1] == 'p' && std::isdigit(s[2]))
{
    op.type = OperandType::PREDICATE;
    // Parse predicate index...
}
```

### 3. Parser: Recognize Labels
**File:** `src/parser/parser.cpp` line 880

**Problem:** Labels like `loop_start` didn't match any pattern and became UNKNOWN

**Fix:** Add label recognition:
```cpp
if (!s.empty() && (std::isalpha(s[0]) || s[0] == '_'))
{
    op.type = OperandType::LABEL;
    op.labelName = s;
}
```

### 4. Executor: Handle LABEL Operands in BRA
**File:** `src/execution/executor.cpp` line 1047

**Fix:** Added label resolution using `resolveLabel()`:
```cpp
if (instr.sources[0].type == OperandType::LABEL) {
    if (resolveLabel(instr.sources[0].labelName, target)) {
        targetResolved = true;
    }
}
```

### 5. Executor: SETP Use getSourceValue()
**File:** `src/execution/executor.cpp` line 1368

**Problem:** SETP tried to read both operands as registers, but might have immediates

**Fix:** Use `getSourceValue()` helper:
```cpp
int64_t src1_val = getSourceValue(instr.sources[0]);
int64_t src2_val = getSourceValue(instr.sources[1]);
```

### 6. Executor: Read Predicates from RegisterBank
**File:** `src/execution/executor.cpp` line 374

**Problem:** Predicate evaluation read from PredicateHandler, but SETP writes to RegisterBank

**Fix:** Read directly from RegisterBank:
```cpp
bool predicateRegValue = m_registerBank->readPredicate(instr.predicateIndex);
bool shouldExecute = (instr.predicateValue == predicateRegValue);
```

### 7. Executor: Use Single Warp (WORKAROUND)
**File:** `src/execution/executor.cpp` line 20

**Problem:** 4 warps all sharing same register file caused each instruction to execute 4 times

**Temporary Fix:** Use 1 warp until per-thread register files are implemented:
```cpp
m_warpScheduler = std::make_unique<WarpScheduler>(1, 32);
```

**TODO:** Implement proper SIMT with per-thread register files

### 8. Type System: Added LABEL Operand Type
**File:** `include/instruction_types.hpp`

Added to `OperandType` enum:
```cpp
enum class OperandType {
    REGISTER,
    IMMEDIATE,
    MEMORY,
    PREDICATE,
    LABEL,      // NEW: for branch targets
    UNKNOWN
};
```

Added to `Operand` struct:
```cpp
struct Operand {
    OperandType type;
    union { /* ... */ };
    // ...
    std::string labelName;  // NEW: for LABEL type
};
```

### 9. Decoder: Also Updated (for completeness)
**File:** `src/decoder/decoder.cpp`

Updated to handle predicate destinations and label operands (matches parser changes)

## Test Results

### Before All Fixes
```
Invalid SETP instruction format (many times)
Invalid BRA instruction format (many times)
branches = 0
result = 40
❌ FAILED
```

### After All Fixes
```
No errors
branches > 0 ✅
result = 50 ✅
✅ PASSED
```

## Architecture Notes

### Current Limitations
1. **Shared Register File**: All threads share one register file
   - This works for single-threaded kernels
   - WRONG for multi-threaded SIMT execution
   
2. **Workaround**: Use 1 warp only
   - Sufficient for basic tests
   - Not scalable for real GPU simulation

### Future Work
1. **Per-Thread Register Files**
   - Each thread needs its own register state
   - Register file should be indexed by (warp, thread, register)
   
2. **Proper SIMT Execution**
   - Track active mask per warp
   - Handle divergence correctly
   - Implement reconvergence

3. **Thread Configuration**
   - Parse kernel launch parameters (grid, block sizes)
   - Initialize appropriate number of warps/threads

## Files Modified
1. `include/instruction_types.hpp` - Type system
2. `src/parser/parser.cpp` - Instruction parsing
3. `src/decoder/decoder.cpp` - Operand decoding
4. `src/execution/executor.cpp` - Instruction execution

## Testing
Run: `./build/tests/smoke_test --gtest_filter=SystemSmokeTest.TestControlFlowExecution`

Expected output:
```
[  PASSED  ] SystemSmokeTest.TestControlFlowExecution
```
