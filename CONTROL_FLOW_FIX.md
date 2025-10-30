# Control Flow Bug Fixes

## Problem Summary
The control flow test was failing with:
- "Invalid SETP instruction format" errors (FIXED)
- "Invalid BRA instruction format" errors - sources.size()=0 (FIXED)
- Branch counter was 0 (no branches counted)
- Loop executed 4 times instead of 5 (result was 40 instead of 50)

## Root Causes

### 1. Parser Treating Branch Target as Destination (CRITICAL BUG)
**Location:** `src/parser/parser.cpp` - `parseInstruction()` function at line 336

**Problem:** The parser assumed ALL instructions follow the pattern: `opcode dest, src1, src2, ...`
This meant for `bra loop_start`, it treated `loop_start` as the **destination** instead of a **source**.
This left the `sources` array **empty**, causing "Invalid BRA instruction format" errors.

**Fix:** Added special handling for instructions with no destination:
```cpp
if (instr.opcode == "bra" || instr.opcode == "call" || 
    instr.opcode == "ret" || instr.opcode == "exit")
{
    // All operands are sources for branch/call/return instructions
    for (const auto& op : operands)
    {
        instr.sources.push_back(op);
    }
}
```

### 2. Parser Not Recognizing Predicate Registers
**Location:** `src/parser/parser.cpp` - `parseOperand()` function

**Problem:** When parsing `%p1`, the function checked for `%` first and immediately returned `OperandType::REGISTER`. It never checked if it was specifically a predicate register (`%p`).

**Fix:** Added a check for `%p` pattern BEFORE the general `%` check:
```cpp
if (s.size() >= 3 && s[1] == 'p' && std::isdigit(s[2]))
{
    op.type = OperandType::PREDICATE;
    // Parse predicate index...
}
```

### 2. Parser Not Recognizing Labels
**Location:** `src/parser/parser.cpp` - `parseOperand()` function

**Problem:** Branch targets like `loop_start`, `loop_body`, `loop_end` didn't match any pattern (not registers, not immediates, not memory), so they fell through to `OperandType::UNKNOWN`.

**Fix:** Added label recognition at the end of `parseOperand()`:
```cpp
// If nothing else matched, treat it as a label (for branch targets)
if (!s.empty() && (std::isalpha(s[0]) || s[0] == '_'))
{
    op.type = OperandType::LABEL;
    op.labelName = s;
    return op;
}
```

### 3. Executor Not Handling LABEL Operands in BRA
**Location:** `src/execution/executor.cpp` - `executeBRA()` function

**Problem:** Branch instruction only handled `IMMEDIATE` and `REGISTER` operand types, not `LABEL`.

**Fix:** Added label resolution:
```cpp
if (instr.sources[0].type == OperandType::LABEL) {
    if (resolveLabel(instr.sources[0].labelName, target)) {
        targetResolved = true;
    }
}
```

### 4. Executor Not Handling Immediate Operands in SETP
**Location:** `src/execution/executor.cpp` - `executeSETP()` function

**Problem:** SETP tried to read both operands directly from registers, but some operands might be immediates (like `setp.lt.s32 %p1, %r4, 5`).

**Fix:** Use `getSourceValue()` helper which handles both registers and immediates:
```cpp
int64_t src1_val = getSourceValue(instr.sources[0]);
int64_t src2_val = getSourceValue(instr.sources[1]);
```

### 5. Predicate Guard Evaluation Reading Wrong State
**Location:** `src/execution/executor.cpp` - `executeDecodedInstruction()` function

**Problem:** Instructions with predicate guards (like `@%p1 bra loop_body`) were calling `m_predicateHandler->evaluatePredicate()`, which read from PredicateHandler's internal state. But SETP writes to RegisterBank, so they were out of sync.

**Fix:** Read predicate value directly from RegisterBank:
```cpp
bool predicateRegValue = m_registerBank->readPredicate(instr.predicateIndex);
bool shouldExecute = (instr.predicateValue == predicateRegValue);
```

## Additional Changes

### Added to `include/instruction_types.hpp`:
- Added `LABEL` to `OperandType` enum
- Added `std::string labelName` field to `Operand` struct (outside union)

### Modified in `src/decoder/decoder.cpp`:
- Updated to recognize predicate destinations (`%p` registers)
- Updated to treat unknown operands as labels

## Testing
After these fixes, the control flow test should:
1. Properly parse SETP instructions with predicate destinations
2. Properly parse BRA instructions with label targets
3. Execute branches and count them in performance counters
4. Run the loop 5 times and produce result of 50

## Files Modified
1. `include/instruction_types.hpp` - Added LABEL type and labelName field
2. `src/parser/parser.cpp` - Fixed parseOperand() to recognize predicates and labels
3. `src/decoder/decoder.cpp` - Added predicate and label support
4. `src/execution/executor.cpp` - Fixed executeBRA(), executeSETP(), and predicate evaluation
