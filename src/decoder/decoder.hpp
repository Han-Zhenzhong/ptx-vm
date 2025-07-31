#ifndef PTX_DECODER_HPP
#define PTX_DECODER_HPP

#include <vector>
#include "parser/parser.hpp"
#include "include/instruction_types.hpp"

// Forward declaration of VM core classes
class VMCore;

class Decoder {
public:
    // Constructor/destructor
    Decoder(VMCore* vmCore);
    ~Decoder();

    // Decode a collection of PTX instructions
    bool decodeInstructions(const std::vector<PTXInstruction>& ptInstructions);

    // Get decoded instruction count
    size_t getDecodedInstructionCount() const;

    // Get decoded instructions
    const std::vector<DecodedInstruction>& getDecodedInstructions() const;

private:
    // Decode instructions
    bool decodeInstruction(const std::string& instruction, DecodedInstruction& decoded);
    
    // Instruction decoding methods
    bool decodeArithmeticInstruction(const std::string& instruction, DecodedInstruction& decoded);
    bool decodeBranchInstruction(const std::string& instruction, DecodedInstruction& decoded);
    bool decodeLoadStoreInstruction(const std::string& instruction, DecodedInstruction& decoded);
    bool decodeSynchronizationInstruction(const std::string& instruction, DecodedInstruction& decoded);
    bool decodeMemoryBarrierInstruction(const std::string& instruction, DecodedInstruction& decoded);
    
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // PTX_DECODER_HPP