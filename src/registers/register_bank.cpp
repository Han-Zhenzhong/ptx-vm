#include "register_bank.hpp"
#include <stdexcept>

RegisterBank::RegisterBank() : 
    m_numRegisters(0), 
    m_numPredicateRegisters(0) {
    // Default constructor
}

RegisterBank::~RegisterBank() {
    // Default destructor
}

bool RegisterBank::initialize(size_t numRegisters) {
    try {
        // Initialize general purpose registers
        m_registers.resize(numRegisters, 0);
        m_numRegisters = numRegisters;
        
        // Initialize predicate registers (typically 8 predicate registers in PTX)
        m_predicateRegisters.resize(8, false);
        m_numPredicateRegisters = 8;
        
        return true;
    } catch (const std::bad_alloc&) {
        // Memory allocation failed
        return false;
    }
}

uint64_t RegisterBank::readRegister(size_t registerIndex) const {
    if (registerIndex >= m_numRegisters) {
        throw std::out_of_range("Register index out of range");
    }
    
    return m_registers[registerIndex];
}

void RegisterBank::writeRegister(size_t registerIndex, uint64_t value) {
    if (registerIndex >= m_numRegisters) {
        throw std::out_of_range("Register index out of range");
    }
    
    m_registers[registerIndex] = value;
}

bool RegisterBank::readPredicate(size_t predicateIndex) const {
    if (predicateIndex >= m_numPredicateRegisters) {
        throw std::out_of_range("Predicate register index out of range");
    }
    
    return m_predicateRegisters[predicateIndex];
}

void RegisterBank::writePredicate(size_t predicateIndex, bool value) {
    if (predicateIndex >= m_numPredicateRegisters) {
        throw std::out_of_range("Predicate register index out of range");
    }
    
    m_predicateRegisters[predicateIndex] = value;
}

size_t RegisterBank::getNumRegisters() const {
    return m_numRegisters;
}

size_t RegisterBank::getNumPredicateRegisters() const {
    return m_numPredicateRegisters;
}