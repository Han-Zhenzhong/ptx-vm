#include "parser/parser.hpp"
#include <iostream>

int main() {
    PTXParser parser;
    
    // Test simple PTX with store instructions
    std::string ptx = R"(
.version 6.0
.target sm_50
.address_size 64

.entry test_kernel (.param .u64 ptr)
{
    .reg .s32 %r<10>;
    
    ld.param.u64 %r0, [ptr];
    mov.s32 %r3, 49;
    mov.s32 %r4, 35;
    
    st.global.s32 [%r0], %r3;
    st.global.s32 [%r0+4], %r4;
    st.global.s32 [%r0+8], %r3;
    
    exit;
}
)";
    
    if (!parser.parseString(ptx)) {
        std::cerr << "Parse failed: " << parser.getErrorMessage() << std::endl;
        return 1;
    }
    
    std::cout << "Parse successful!" << std::endl;
    const auto& instructions = parser.getInstructions();
    std::cout << "Total instructions: " << instructions.size() << std::endl;
    
    for (size_t i = 0; i < instructions.size(); i++) {
        const auto& instr = instructions[i];
        std::cout << "Instruction " << i << ": type=" << static_cast<int>(instr.type);
        if (instr.dest.type == OperandType::MEMORY) {
            std::cout << ", dest=MEMORY, isIndirect=" << instr.dest.isIndirect
                      << ", registerIndex=" << instr.dest.registerIndex
                      << ", address=" << instr.dest.address;
        }
        std::cout << std::endl;
    }
    
    return 0;
}
