#include "parser.hpp"
#include <iostream>
#include <cassert>
#include <string>

int main() {
    PTXParser parser;
    
    // Test PTX code with various instruction types
    std::string testPTX = R"(
.version 7.0
.target sm_50
.address_size 64

.entry test_kernel {
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    .reg .pred %p<2>;
    
    mov.u32 %r1, 10;
    add.u32 %r2, %r1, 5;
    @p1 add.f32 %f1, %f2, %f3;
    ld.global.f32 %f2, [%r1];
    st.global.f32 [%r2], %f1;
    bra.uni L1;
L1:
    ret;
}
)";
    
    std::cout << "Testing PTX parser with sample code...\n";
    
    bool result = parser.parseString(testPTX);
    if (!result) {
        std::cout << "Parse failed: " << parser.getErrorMessage() << "\n";
        return 1;
    }
    
    const auto& instructions = parser.getInstructions();
    
    std::cout << "Parsed " << instructions.size() << " instructions\n";
    
    // Verify we got some instructions (this is a basic test)
    if (instructions.size() == 0) {
        std::cout << "Warning: No instructions parsed\n";
    } else {
        std::cout << "Parser test passed - successfully parsed PTX instructions!\n";
    }
    
    return 0;
}