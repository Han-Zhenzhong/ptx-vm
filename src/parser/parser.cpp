#include "parser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <regex>
#include <unordered_map>

// Private implementation class
class PTXParser::Impl {
public:
    Impl() {
        // Initialize default state
        m_errorMessage = "";
    }
    
    ~Impl() = default;

    // Parse a PTX file
    bool parseFile(const std::string& filename) {
        // Open file
        std::ifstream file(filename);
        if (!file.is_open()) {
            m_errorMessage = "Failed to open file: " + filename;
            return false;
        }
        
        // Read file contents
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();
        
        // Parse contents
        return parseString(buffer.str());
    }

    // Parse PTX code from string
    bool parseString(const std::string& ptxCode) {
        // Clear previous instructions
        m_instructions.clear();
        m_errorMessage = "";
        
        // Split code into lines
        std::istringstream iss(ptxCode);
        std::string line;
        size_t lineNumber = 0;
        
        while (std::getline(iss, line)) {
            lineNumber++;
            // Trim whitespace first
            line = trim(line);
            // Skip empty lines and comments (including lines with leading spaces before //)
            if (line.empty() || line[0] == '#' || (line.length() >= 2 && line[0] == '/' && line[1] == '/')) {
                continue;
            }
            if (line.substr(0, 2) == "//") {
                continue;
            }
            // Skip if still empty (redundant, but safe)
            if (line.empty()) {
                continue;
            }
            // Try to parse the line as an instruction
            PTXInstruction ptInstruction;
            if (parsePTXInstruction(line, ptInstruction)) {
                // Convert PTXInstruction to DecodedInstruction
                DecodedInstruction instruction = {};
                // 1. opcode 映射
                instruction.type = opcodeFromString(ptInstruction.opcode);
                // 2. 目标操作数
                if (!ptInstruction.dest.empty()) {
                    instruction.dest = parseOperand(ptInstruction.dest);
                }
                // 3. 源操作数
                for (const auto& src : ptInstruction.sources) {
                    instruction.sources.push_back(parseOperand(src));
                }
                // 4. 修饰符
                instruction.modifiers = encodeModifiers(ptInstruction.modifiers);
                // 5. 谓词
                instruction.hasPredicate = !ptInstruction.predicate.empty();
                if (instruction.hasPredicate) {
                    // 假设谓词格式为p0/p1等
                    if (ptInstruction.predicate[0] == '!') {
                        instruction.predicateValue = false;
                        instruction.predicateIndex = std::stoi(ptInstruction.predicate.substr(2));
                    } else {
                        instruction.predicateValue = true;
                        instruction.predicateIndex = std::stoi(ptInstruction.predicate.substr(1));
                    }
                }
                m_instructions.push_back(instruction);
            }
        }
        
        return true;
    }

    // --- 辅助函数 ---
    // 1. opcode字符串到InstructionTypes映射
    InstructionTypes opcodeFromString(const std::string& op) {
        static const std::unordered_map<std::string, InstructionTypes> table = {
            {"add", InstructionTypes::ADD},
            {"sub", InstructionTypes::SUB},
            {"mul", InstructionTypes::MUL},
            {"div", InstructionTypes::DIV},
            {"and", InstructionTypes::AND},
            {"or", InstructionTypes::OR},
            {"xor", InstructionTypes::XOR},
            {"not", InstructionTypes::NOT},
            {"shl", InstructionTypes::SHL},
            {"shr", InstructionTypes::SHR},
            {"neg", InstructionTypes::NEG},
            {"abs", InstructionTypes::ABS},
            {"bra", InstructionTypes::BRA},
            {"jump", InstructionTypes::JUMP},
            {"call", InstructionTypes::CALL},
            {"ret", InstructionTypes::RET},
            {"sync", InstructionTypes::SYNC},
            {"membar", InstructionTypes::MEMBAR},
            {"ld", InstructionTypes::LD},
            {"st", InstructionTypes::ST},
            {"mov", InstructionTypes::MOV},
            {"cmov", InstructionTypes::CMOV},
            {"ld.param", InstructionTypes::LD_PARAM},
            {"st.param", InstructionTypes::ST_PARAM},
            {"nop", InstructionTypes::NOP},
            {"barrier", InstructionTypes::BARRIER},
        };
        auto it = table.find(op);
        if (it != table.end()) return it->second;
        // 支持如ld.param/st.param等复合指令
        if (op == "ld" || op == "ld.param") return InstructionTypes::LD_PARAM;
        if (op == "st" || op == "st.param") return InstructionTypes::ST_PARAM;
        return InstructionTypes::MAX_INSTRUCTION_TYPE;
    }

    // 2. 字符串到Operand解析
    Operand parseOperand(const std::string& s) {
        Operand operand = {};
        operand.isAddress = false;
        operand.isIndirect = false;
        std::string str = s;
        // 判断内存寻址
        if (!str.empty() && str.front() == '[' && str.back() == ']') {
            operand.type = OperandType::MEMORY;
            operand.isAddress = true;
            // 去除[]
            str = str.substr(1, str.size() - 2);
            // 支持[%r0+4]等简单表达式，暂存为address=0
            operand.address = 0;
        } else if (!str.empty() && str[0] == '%') {
            // 判断寄存器
            operand.type = OperandType::REGISTER;
            // %r0, %f1, %p2等
            size_t idx = 2;
            if (str.size() > 2 && std::isdigit(str[2])) {
                idx = 2;
            } else if (str.size() > 3 && std::isdigit(str[3])) {
                idx = 3;
            }
            operand.registerIndex = std::stoi(str.substr(idx));
        } else if (!str.empty() && (std::isdigit(str[0]) || (str[0] == '-' && str.size() > 1 && std::isdigit(str[1])))) {
            // 立即数
            operand.type = OperandType::IMMEDIATE;
            operand.immediateValue = std::stoll(str);
        } else if (!str.empty() && str[0] == 'p') {
            // 谓词寄存器
            operand.type = OperandType::PREDICATE;
            operand.predicateIndex = std::stoi(str.substr(1));
        } else {
            operand.type = OperandType::UNKNOWN;
        }
        return operand;
    }

    // 3. 修饰符编码（简单实现：每个修饰符hash后或到一起）
    uint32_t encodeModifiers(const std::vector<std::string>& mods) {
        uint32_t result = 0;
        for (const auto& m : mods) {
            for (char c : m) {
                result ^= (uint32_t)c;
                result = (result << 1) | (result >> 31);
            }
        }
        return result;
    }

    // Get parsed instructions
    const std::vector<DecodedInstruction>& getInstructions() const {
        return m_instructions;
    }

    // Get error message if parsing failed
    const std::string& getErrorMessage() const {
        return m_errorMessage;
    }

private:
    // Helper function to trim all leading and trailing whitespace (space, tab, etc.)
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\n\r\f\v");
        if (first == std::string::npos) {
            return "";
        }
        size_t last = str.find_last_not_of(" \t\n\r\f\v");
        return str.substr(first, (last - first + 1));
    }
    
    // Parse a single PTX instruction line into a PTXInstruction struct
    // 修复点：
    // 1. 支持行尾注释（去除 // 及其后内容）
    // 2. 支持 trim 时去除制表符和所有空白字符
    // 3. 支持立即数、负号、结构体成员等操作数
    // 4. 支持括号嵌套的 splitOperands
    // 5. 支持 label 行只有一个字符时不越界
    bool parsePTXInstruction(const std::string& line, PTXInstruction& instruction) {
        // 去除行尾注释
        std::string codeLine = line;
        size_t commentPos = codeLine.find("//");
        if (commentPos != std::string::npos) {
            codeLine = codeLine.substr(0, commentPos);
        }
        std::string trimmedLine = trim(codeLine);
        if (trimmedLine.empty()) return false;
        // 跳过标签和指令
        if (trimmedLine.size() == 1 && trimmedLine[0] == ':') return false;
        if (trimmedLine.back() == ':' || trimmedLine[0] == '.') {
            return false;
        }
        // 支持复合操作码和类型修饰符分离（如ld.param.u64 => opcode=ld.param, modifier=u64）
        std::regex instructionRegex(R"(^(@\w+)?\s*([\w]+(?:\.[\w]+)*?)\.(\w+)\s+(.+)$)");
        std::smatch matches;
        if (std::regex_search(trimmedLine, matches, instructionRegex) && matches.size() > 4) {
            // 谓词
            if (matches[1].matched) {
                instruction.predicate = matches[1].str().substr(1);
            }
            // 操作码（如ld.param）
            instruction.opcode = matches[2].str();
            // 修饰符（如u64）
            if (matches[3].matched) {
                std::string modifiersStr = matches[3].str();
                if (!modifiersStr.empty()) {
                    instruction.modifiers.push_back(modifiersStr);
                }
            }
            // 操作数
            std::string operandsStr = matches[4].str();
            std::vector<std::string> operands = splitOperands(operandsStr);
            if (!operands.empty()) {
                instruction.dest = operands[0];
                for (size_t i = 1; i < operands.size(); ++i) {
                    instruction.sources.push_back(operands[i]);
                }
            }
            return true;
        } else {
            // 无操作数指令
            std::regex simpleRegex(R"(^(@\w+)?\s*([\w]+(?:\.[\w]+)*?)\.(\w+)$)");
            if (std::regex_search(trimmedLine, matches, simpleRegex) && matches.size() > 2) {
                if (matches[1].matched) {
                    instruction.predicate = matches[1].str().substr(1);
                }
                instruction.opcode = matches[2].str();
                if (matches[3].matched) {
                    std::string modifiersStr = matches[3].str();
                    if (!modifiersStr.empty()) {
                        instruction.modifiers.push_back(modifiersStr);
                    }
                }
                return true;
            }
            // 没有修饰符的情况
            std::regex noModRegex(R"(^(@\w+)?\s*([\w]+(?:\.[\w]+)*)\s+(.+)$)");
            if (std::regex_search(trimmedLine, matches, noModRegex) && matches.size() > 3) {
                if (matches[1].matched) {
                    instruction.predicate = matches[1].str().substr(1);
                }
                instruction.opcode = matches[2].str();
                std::string operandsStr = matches[3].str();
                std::vector<std::string> operands = splitOperands(operandsStr);
                if (!operands.empty()) {
                    instruction.dest = operands[0];
                    for (size_t i = 1; i < operands.size(); ++i) {
                        instruction.sources.push_back(operands[i]);
                    }
                }
                return true;
            }
            // 无操作数无修饰符
            std::regex noModSimpleRegex(R"(^(@\w+)?\s*([\w]+(?:\.[\w]+)*)$)");
            if (std::regex_search(trimmedLine, matches, noModSimpleRegex) && matches.size() > 2) {
                if (matches[1].matched) {
                    instruction.predicate = matches[1].str().substr(1);
                }
                instruction.opcode = matches[2].str();
                return true;
            }
        }
        return false;
    }
    
    // Split operands by comma, but respect all types of brackets ([], (), {})
    std::vector<std::string> splitOperands(const std::string& operandsStr) {
        std::vector<std::string> operands;
        std::string currentOperand;
        int round = 0, square = 0, curly = 0;
        for (size_t i = 0; i < operandsStr.size(); ++i) {
            char c = operandsStr[i];
            if (c == ',' && round == 0 && square == 0 && curly == 0) {
                std::string trimmed = trim(currentOperand);
                if (!trimmed.empty()) {
                    operands.push_back(trimmed);
                }
                currentOperand.clear();
            } else {
                if (c == '(') round++;
                if (c == ')') round--;
                if (c == '[') square++;
                if (c == ']') square--;
                if (c == '{') curly++;
                if (c == '}') curly--;
                currentOperand += c;
            }
        }
        std::string trimmed = trim(currentOperand);
        if (!trimmed.empty()) {
            operands.push_back(trimmed);
        }
        return operands;
    }
    
    // Parsed instructions
    std::vector<DecodedInstruction> m_instructions;
    
    // Error message
    std::string m_errorMessage;
};

PTXParser::PTXParser() : pImpl(std::make_unique<Impl>()) {}

PTXParser::~PTXParser() = default;

bool PTXParser::parseFile(const std::string& filename) {
    return pImpl->parseFile(filename);
}

bool PTXParser::parseString(const std::string& ptxCode) {
    return pImpl->parseString(ptxCode);
}

const std::vector<DecodedInstruction>& PTXParser::getInstructions() const {
    return pImpl->getInstructions();
}

const std::string& PTXParser::getErrorMessage() const {
    return pImpl->getErrorMessage();
}

// Factory functions
extern "C" {
    PTXParser* createPTXParser() {
        return new PTXParser();
    }
    
    void destroyPTXParser(PTXParser* parser) {
        delete parser;
    }
}