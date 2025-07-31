#ifndef REGISTER_BANK_HPP
#define REGISTER_BANK_HPP

#include <cstdint>
#include <vector>

class RegisterBank {
public:
    // Constructor/destructor
    RegisterBank();
    ~RegisterBank();

    // Initialize register bank with specified number of registers
    bool initialize(size_t numRegisters = 32);

    // Read a register value
    uint64_t readRegister(size_t registerIndex) const;

    // Write a register value
    void writeRegister(size_t registerIndex, uint64_t value);

    // Read a predicate register
    bool readPredicate(size_t predicateIndex) const;

    // Write a predicate register
    void writePredicate(size_t predicateIndex, bool value);

    // Get number of registers
    size_t getNumRegisters() const;

    // Get number of predicate registers
    size_t getNumPredicateRegisters() const;

private:
    // General purpose registers
    std::vector<uint64_t> m_registers;
    
    // Predicate registers
    std::vector<bool> m_predicateRegisters;
    
    // Configuration
    size_t m_numRegisters;
    size_t m_numPredicateRegisters;
};

#endif // REGISTER_BANK_HPP