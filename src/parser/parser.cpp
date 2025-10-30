#include "parser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <cctype>
#include <algorithm>

// PTXInstruction is defined in instruction_types.hpp (included via parser.hpp)

// PTXParser::Impl - 私有实现类
class PTXParser::Impl
{
public:
    Impl() = default;
    ~Impl() = default;

    bool parseFile(const std::string &filename);
    bool parseString(const std::string &ptxCode);
    const PTXProgram &getProgram() const { return m_program; }
    const std::vector<DecodedInstruction> &getInstructions() const { return m_program.instructions; }
    const std::string &getErrorMessage() const { return m_errorMessage; }

private:
    void preprocessLines(const std::string &ptxCode);
    bool firstPass();
    bool secondPass();
    bool parseMetadata(const std::string &line);
    PTXFunction *parseFunctionDeclaration(const std::string &line, size_t &lineIndex);
    std::string extractFunctionName(const std::string &declaration);
    std::vector<PTXParameter> parseParameters(const std::string &declaration);
    PTXRegisterDeclaration parseRegisterDeclaration(const std::string &line);
    bool parseInstruction(const std::string &line, PTXInstruction &instr);
    DecodedInstruction convertToDecoded(const PTXInstruction &ptxInstr);
    void buildSymbolTable();

    std::string trim(const std::string &str);
    std::string extractValue(const std::string &line, const std::string &directive);
    std::vector<std::string> split(const std::string &str, char delimiter);
    std::vector<std::string> splitOperands(const std::string &operandsStr);
    size_t getTypeSize(const std::string &type);
    InstructionTypes opcodeToInstructionType(const std::string &opcode, const std::vector<std::string>& modifiers = {});
    Operand parseOperand(const std::string &str);

private:
    PTXProgram m_program;
    std::string m_errorMessage;
    std::vector<std::string> m_lines;
    PTXFunction* m_currentFunction = nullptr;  // Track current function being parsed
};

// 实现部分将在下一个文件中继续...
// 占位符实现

bool PTXParser::Impl::parseFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        m_errorMessage = "Failed to open file: " + filename;
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return parseString(buffer.str());
}

bool PTXParser::Impl::parseString(const std::string &ptxCode)
{
    m_program = PTXProgram();
    m_errorMessage = "";
    m_lines.clear();
    preprocessLines(ptxCode);
    if (!firstPass())
        return false;
    if (!secondPass())
        return false;
    buildSymbolTable();
    return true;
}

void PTXParser::Impl::preprocessLines(const std::string &ptxCode)
{
    std::istringstream iss(ptxCode);
    std::string line;
    while (std::getline(iss, line))
    {
        size_t commentPos = line.find("//");
        if (commentPos != std::string::npos)
        {
            line = line.substr(0, commentPos);
        }
        line = trim(line);
        
        // Remove trailing semicolon if present
        if (!line.empty() && line.back() == ';')
        {
            line = line.substr(0, line.size() - 1);
            line = trim(line);  // Trim again after removing semicolon
        }
        
        if (!line.empty())
        {
            m_lines.push_back(line);
        }
    }
}

bool PTXParser::Impl::firstPass()
{
    PTXFunction *currentFunction = nullptr;
    size_t instructionCount = 0;
    for (size_t i = 0; i < m_lines.size(); ++i)
    {
        const std::string &line = m_lines[i];
        if (parseMetadata(line))
            continue;
        if (line.find(".entry") == 0 || line.find(".func") == 0)
        {
            currentFunction = parseFunctionDeclaration(line, i);
            if (currentFunction)
            {
                currentFunction->startInstructionIndex = instructionCount;
            }
            continue;
        }
        if (line == "{")
            continue;
        if (line == "}")
        {
            if (currentFunction)
            {
                currentFunction->endInstructionIndex = instructionCount > 0 ? instructionCount - 1 : 0;
            }
            currentFunction = nullptr;
            continue;
        }
        if (currentFunction)
        {
            if (line.find(".reg") == 0)
            {
                PTXRegisterDeclaration regDecl = parseRegisterDeclaration(line);
                if (!regDecl.type.empty())
                {
                    currentFunction->registerDeclarations.push_back(regDecl);
                }
                continue;
            }
            if (line.back() == ':')
            {
                std::string labelName = line.substr(0, line.size() - 1);
                currentFunction->localLabels[labelName] = instructionCount;
                continue;
            }
            if (line[0] != '.')
            {
                instructionCount++;
            }
        }
    }
    return true;
}

bool PTXParser::Impl::secondPass()
{
    bool inFunctionBody = false;
    m_currentFunction = nullptr;
    
    for (size_t i = 0; i < m_lines.size(); ++i)
    {
        const std::string& line = m_lines[i];
        
        // Check for function declaration (before opening brace)
        if (!inFunctionBody && (line.find(".entry") == 0 || line.find(".func") == 0))
        {
            std::string funcName = extractFunctionName(line);
            // Find this function in the already-parsed functions
            for (auto& func : m_program.functions)
            {
                if (func.name == funcName)
                {
                    m_currentFunction = &func;
                    break;
                }
            }
        }
        
        if (line == "{")
        {
            inFunctionBody = true;
            continue;
        }
        if (line == "}")
        {
            inFunctionBody = false;
            m_currentFunction = nullptr;
            continue;
        }
        if (!inFunctionBody)
            continue;
        if (line[0] == '.' || line.back() == ':')
            continue;
        PTXInstruction ptxInstr;
        if (parseInstruction(line, ptxInstr))
        {
            DecodedInstruction decoded = convertToDecoded(ptxInstr);
            m_program.instructions.push_back(decoded);
        }
    }
    return true;
}

bool PTXParser::Impl::parseMetadata(const std::string &line)
{
    if (line.find(".version") == 0)
    {
        m_program.metadata.version = extractValue(line, ".version");
        return true;
    }
    if (line.find(".target") == 0)
    {
        m_program.metadata.target = extractValue(line, ".target");
        m_program.metadata.debugMode = (line.find("debug") != std::string::npos);
        return true;
    }
    if (line.find(".address_size") == 0)
    {
        std::string sizeStr = extractValue(line, ".address_size");
        if (!sizeStr.empty())
        {
            m_program.metadata.addressSize = std::stoi(sizeStr);
        }
        return true;
    }
    return false;
}

PTXFunction *PTXParser::Impl::parseFunctionDeclaration(const std::string &line, size_t &lineIndex)
{
    PTXFunction func;
    func.isEntry = (line.find(".entry") == 0);
    std::string fullDecl = line;
    while (fullDecl.find(")") == std::string::npos && lineIndex + 1 < m_lines.size())
    {
        lineIndex++;
        fullDecl += " " + m_lines[lineIndex];
    }
    func.name = extractFunctionName(fullDecl);
    func.parameters = parseParameters(fullDecl);
    m_program.functions.push_back(func);
    if (func.isEntry)
    {
        m_program.entryPoints.push_back(m_program.functions.size() - 1);
    }
    return &m_program.functions.back();
}

std::string PTXParser::Impl::extractFunctionName(const std::string &declaration)
{
    std::regex nameRegex(R"(\.(?:entry|func)\s+(?:\([^)]*\)\s+)?(\w+)\s*\()");
    std::smatch matches;
    if (std::regex_search(declaration, matches, nameRegex))
    {
        return matches[1].str();
    }
    return "";
}

std::vector<PTXParameter> PTXParser::Impl::parseParameters(const std::string &declaration)
{
    std::vector<PTXParameter> params;
    size_t start = declaration.find('(');
    size_t end = declaration.rfind(')');
    if (start == std::string::npos || end == std::string::npos)
        return params;
    std::string paramStr = declaration.substr(start + 1, end - start - 1);
    std::regex paramRegex(R"(\.param\s+(\.[\w]+)\s+(\w+))");
    std::smatch matches;
    std::string::const_iterator searchStart(paramStr.cbegin());
    size_t offset = 0;
    while (std::regex_search(searchStart, paramStr.cend(), matches, paramRegex))
    {
        PTXParameter param;
        param.type = matches[1].str();
        param.name = matches[2].str();
        param.offset = offset;
        param.size = getTypeSize(param.type);
        param.isPointer = (param.type == ".u64" || param.type == ".s64");
        params.push_back(param);
        offset += param.size;
        searchStart = matches.suffix().first;
    }
    return params;
}

PTXRegisterDeclaration PTXParser::Impl::parseRegisterDeclaration(const std::string &line)
{
    PTXRegisterDeclaration decl;
    std::regex regRegex(R"(\.reg\s+(\.[\w]+)\s+%(\w+)<(\d+)>)");
    std::smatch matches;
    if (std::regex_search(line, matches, regRegex))
    {
        decl.type = matches[1].str();
        decl.baseRegister = matches[2].str();
        decl.startIndex = 0;
        decl.count = std::stoi(matches[3].str());
    }
    return decl;
}

bool PTXParser::Impl::parseInstruction(const std::string &line, PTXInstruction &instr)
{
    std::string remaining = line;
    if (remaining[0] == '@')
    {
        size_t spacePos = remaining.find(' ');
        if (spacePos != std::string::npos)
        {
            instr.predicate = remaining.substr(1, spacePos - 1);
            remaining = trim(remaining.substr(spacePos + 1));
        }
    }
    size_t spacePos = remaining.find(' ');
    std::string opcodeWithMods = (spacePos != std::string::npos) ? remaining.substr(0, spacePos) : remaining;
    std::vector<std::string> parts = split(opcodeWithMods, '.');
    if (parts.empty())
        return false;
    instr.opcode = parts[0];
    for (size_t i = 1; i < parts.size(); ++i)
    {
        instr.modifiers.push_back("." + parts[i]);
    }
    if (spacePos != std::string::npos)
    {
        std::string operandsStr = trim(remaining.substr(spacePos + 1));
        std::vector<std::string> operands = splitOperands(operandsStr);
        if (!operands.empty())
        {
            instr.dest = operands[0];
            for (size_t i = 1; i < operands.size(); ++i)
            {
                instr.sources.push_back(operands[i]);
            }
        }
    }
    return true;
}

DecodedInstruction PTXParser::Impl::convertToDecoded(const PTXInstruction &ptxInstr)
{
    DecodedInstruction decoded = {};
    decoded.type = opcodeToInstructionType(ptxInstr.opcode, ptxInstr.modifiers);
    
    // Parse data type from modifiers
    decoded.dataType = DataType::U32; // default
    for (const auto& mod : ptxInstr.modifiers) {
        if (mod == ".s8") decoded.dataType = DataType::S8;
        else if (mod == ".s16") decoded.dataType = DataType::S16;
        else if (mod == ".s32") decoded.dataType = DataType::S32;
        else if (mod == ".s64") decoded.dataType = DataType::S64;
        else if (mod == ".u8") decoded.dataType = DataType::U8;
        else if (mod == ".u16") decoded.dataType = DataType::U16;
        else if (mod == ".u32") decoded.dataType = DataType::U32;
        else if (mod == ".u64") decoded.dataType = DataType::U64;
        else if (mod == ".f16") decoded.dataType = DataType::F16;
        else if (mod == ".f32") decoded.dataType = DataType::F32;
        else if (mod == ".f64") decoded.dataType = DataType::F64;
    }
    
    // Parse CVT instruction types (cvt.dstType.srcType)
    // For CVT, modifiers are in format: [".cvt", ".dstType", ".srcType"]
    if (ptxInstr.opcode == "cvt" && ptxInstr.modifiers.size() >= 2) {
        // First modifier is destination type, second is source type
        auto parseType = [](const std::string& mod) -> DataType {
            if (mod == ".s8") return DataType::S8;
            if (mod == ".s16") return DataType::S16;
            if (mod == ".s32") return DataType::S32;
            if (mod == ".s64") return DataType::S64;
            if (mod == ".u8") return DataType::U8;
            if (mod == ".u16") return DataType::U16;
            if (mod == ".u32") return DataType::U32;
            if (mod == ".u64") return DataType::U64;
            if (mod == ".f16") return DataType::F16;
            if (mod == ".f32") return DataType::F32;
            if (mod == ".f64") return DataType::F64;
            return DataType::U32;
        };
        
        decoded.dstType = parseType(ptxInstr.modifiers[0]);
        decoded.srcType = parseType(ptxInstr.modifiers[1]);
    }
    
    // Parse comparison operator from modifiers (for setp)
    decoded.compareOp = CompareOp::EQ; // default
    for (const auto& mod : ptxInstr.modifiers) {
        if (mod == ".eq") decoded.compareOp = CompareOp::EQ;
        else if (mod == ".ne") decoded.compareOp = CompareOp::NE;
        else if (mod == ".lt") decoded.compareOp = CompareOp::LT;
        else if (mod == ".le") decoded.compareOp = CompareOp::LE;
        else if (mod == ".gt") decoded.compareOp = CompareOp::GT;
        else if (mod == ".ge") decoded.compareOp = CompareOp::GE;
        else if (mod == ".lo") decoded.compareOp = CompareOp::LO;
        else if (mod == ".ls") decoded.compareOp = CompareOp::LS;
        else if (mod == ".hi") decoded.compareOp = CompareOp::HI;
        else if (mod == ".hs") decoded.compareOp = CompareOp::HS;
    }
    
    if (!ptxInstr.dest.empty())
    {
        decoded.dest = parseOperand(ptxInstr.dest);
    }
    for (const auto &src : ptxInstr.sources)
    {
        decoded.sources.push_back(parseOperand(src));
    }
    if (!ptxInstr.predicate.empty())
    {
        decoded.hasPredicate = true;
        std::string pred = ptxInstr.predicate;
        
        // Handle negation (!)
        if (pred[0] == '!')
        {
            decoded.predicateValue = false;
            pred = pred.substr(1); // Remove '!'
        }
        else
        {
            decoded.predicateValue = true;
        }
        
        // Handle predicate register prefix (%)
        if (!pred.empty() && pred[0] == '%')
        {
            pred = pred.substr(1); // Remove '%'
        }
        
        // Handle predicate register prefix (p)
        if (!pred.empty() && pred[0] == 'p')
        {
            pred = pred.substr(1); // Remove 'p'
        }
        
        // Now extract the predicate index number
        if (!pred.empty())
        {
            decoded.predicateIndex = std::stoi(pred);
        }
    }
    decoded.modifiers = 0;
    for (const auto &mod : ptxInstr.modifiers)
    {
        decoded.modifiers ^= std::hash<std::string>{}(mod);
    }
    return decoded;
}

void PTXParser::Impl::buildSymbolTable()
{
    for (auto &func : m_program.functions)
    {
        m_program.symbolTable.functions[func.name] = func;
        for (auto &param : func.parameters)
        {
            m_program.symbolTable.parameterSymbols[param.name] = &param;
        }
    }
}

std::string PTXParser::Impl::trim(const std::string &str)
{
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string::npos)
        return "";
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, last - first + 1);
}

std::string PTXParser::Impl::extractValue(const std::string &line, const std::string &directive)
{
    size_t pos = line.find(directive);
    if (pos == std::string::npos)
        return "";
    std::string value = line.substr(pos + directive.length());
    return trim(value);
}

std::vector<std::string> PTXParser::Impl::split(const std::string &str, char delimiter)
{
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter))
    {
        if (!item.empty())
        {
            result.push_back(item);
        }
    }
    return result;
}

std::vector<std::string> PTXParser::Impl::splitOperands(const std::string &operandsStr)
{
    std::vector<std::string> operands;
    std::string current;
    int bracketDepth = 0;
    for (char c : operandsStr)
    {
        if (c == '[')
        {
            bracketDepth++;
            current += c;
        }
        else if (c == ']')
        {
            bracketDepth--;
            current += c;
        }
        else if (c == ',' && bracketDepth == 0)
        {
            std::string trimmed = trim(current);
            if (!trimmed.empty())
            {
                operands.push_back(trimmed);
            }
            current.clear();
        }
        else
        {
            current += c;
        }
    }
    std::string trimmed = trim(current);
    if (!trimmed.empty())
    {
        operands.push_back(trimmed);
    }
    return operands;
}

size_t PTXParser::Impl::getTypeSize(const std::string &type)
{
    if (type == ".b8" || type == ".s8" || type == ".u8")
        return 1;
    if (type == ".b16" || type == ".s16" || type == ".u16" || type == ".f16")
        return 2;
    if (type == ".b32" || type == ".s32" || type == ".u32" || type == ".f32")
        return 4;
    if (type == ".b64" || type == ".s64" || type == ".u64" || type == ".f64")
        return 8;
    return 4;
}

InstructionTypes PTXParser::Impl::opcodeToInstructionType(const std::string &opcode, const std::vector<std::string>& modifiers)
{
    // Helper to check if a modifier exists
    auto hasModifier = [&modifiers](const std::string& mod) {
        return std::find(modifiers.begin(), modifiers.end(), mod) != modifiers.end();
    };
    
    // Check for floating-point modifiers
    bool isF32 = hasModifier(".f32");
    bool isF64 = hasModifier(".f64");
    
    // Floating-point and integer instructions
    if (opcode == "add") {
        if (isF32) return InstructionTypes::ADD_F32;
        if (isF64) return InstructionTypes::ADD_F64;
        return InstructionTypes::ADD;
    }
    if (opcode == "sub") {
        if (isF32) return InstructionTypes::SUB_F32;
        if (isF64) return InstructionTypes::SUB_F64;
        return InstructionTypes::SUB;
    }
    if (opcode == "mul") {
        if (isF32) return InstructionTypes::MUL_F32;
        if (isF64) return InstructionTypes::MUL_F64;
        return InstructionTypes::MUL;
    }
    if (opcode == "div") {
        if (isF32) return InstructionTypes::DIV_F32;
        if (isF64) return InstructionTypes::DIV_F64;
        return InstructionTypes::DIV;
    }
    if (opcode == "neg") {
        if (isF32) return InstructionTypes::NEG_F32;
        if (isF64) return InstructionTypes::NEG_F64;
        return InstructionTypes::NEG;
    }
    if (opcode == "abs") {
        if (isF32) return InstructionTypes::ABS_F32;
        if (isF64) return InstructionTypes::ABS_F64;
        return InstructionTypes::ABS;
    }
    
    // Floating-point specific instructions
    if (opcode == "fma") {
        if (isF32) return InstructionTypes::FMA_F32;
        if (isF64) return InstructionTypes::FMA_F64;
        return InstructionTypes::MAX_INSTRUCTION_TYPE;
    }
    if (opcode == "sqrt") {
        if (isF32) return InstructionTypes::SQRT_F32;
        if (isF64) return InstructionTypes::SQRT_F64;
        return InstructionTypes::MAX_INSTRUCTION_TYPE;
    }
    if (opcode == "rsqrt") {
        if (isF32) return InstructionTypes::RSQRT_F32;
        if (isF64) return InstructionTypes::RSQRT_F64;
        return InstructionTypes::MAX_INSTRUCTION_TYPE;
    }
    if (opcode == "min") {
        if (isF32) return InstructionTypes::MIN_F32;
        if (isF64) return InstructionTypes::MIN_F64;
        return InstructionTypes::MAX_INSTRUCTION_TYPE;
    }
    if (opcode == "max") {
        if (isF32) return InstructionTypes::MAX_F32;
        if (isF64) return InstructionTypes::MAX_F64;
        return InstructionTypes::MAX_INSTRUCTION_TYPE;
    }
    
    // Comparison and selection instructions
    if (opcode == "setp") return InstructionTypes::SETP;
    if (opcode == "selp") return InstructionTypes::SELP;
    if (opcode == "set") return InstructionTypes::SET;
    
    // Type conversion
    if (opcode == "cvt") return InstructionTypes::CVT;
    
    // Atomic operations
    if (opcode == "atom") {
        if (hasModifier(".add")) return InstructionTypes::ATOM_ADD;
        if (hasModifier(".sub")) return InstructionTypes::ATOM_SUB;
        if (hasModifier(".exch")) return InstructionTypes::ATOM_EXCH;
        if (hasModifier(".cas")) return InstructionTypes::ATOM_CAS;
        if (hasModifier(".min")) return InstructionTypes::ATOM_MIN;
        if (hasModifier(".max")) return InstructionTypes::ATOM_MAX;
        if (hasModifier(".inc")) return InstructionTypes::ATOM_INC;
        if (hasModifier(".dec")) return InstructionTypes::ATOM_DEC;
        if (hasModifier(".and")) return InstructionTypes::ATOM_AND;
        if (hasModifier(".or")) return InstructionTypes::ATOM_OR;
        if (hasModifier(".xor")) return InstructionTypes::ATOM_XOR;
        return InstructionTypes::MAX_INSTRUCTION_TYPE;
    }
    
    // Integer-only instructions (keep existing logic)
    if (opcode == "rem")
        return InstructionTypes::REM;
    if (opcode == "and")
        return InstructionTypes::AND;
    if (opcode == "or")
        return InstructionTypes::OR;
    if (opcode == "xor")
        return InstructionTypes::XOR;
    if (opcode == "not")
        return InstructionTypes::NOT;
    if (opcode == "shl")
        return InstructionTypes::SHL;
    if (opcode == "shr")
        return InstructionTypes::SHR;
    if (opcode == "bra")
        return InstructionTypes::BRA;
    if (opcode == "call")
        return InstructionTypes::CALL;
    if (opcode == "ret" || opcode == "exit")
        return InstructionTypes::RET;
    if (opcode == "ld") {
        // Check for parameter load
        if (hasModifier(".param"))
            return InstructionTypes::LD_PARAM;
        return InstructionTypes::LD;
    }
    if (opcode == "st") {
        // Check for parameter store
        if (hasModifier(".param"))
            return InstructionTypes::ST_PARAM;
        return InstructionTypes::ST;
    }
    if (opcode == "mov")
        return InstructionTypes::MOV;
    if (opcode == "bar" || opcode == "barrier")
        return InstructionTypes::BARRIER;
    return InstructionTypes::MAX_INSTRUCTION_TYPE;
}

Operand PTXParser::Impl::parseOperand(const std::string &str)
{
    Operand op = {};
    op.isAddress = false;
    op.isIndirect = false;
    std::string s = trim(str);
    
    // Handle empty strings
    if (s.empty()) {
        op.type = OperandType::UNKNOWN;
        return op;
    }
    
    // Handle memory operands with brackets [...]
    if (s.size() >= 2 && s[0] == '[' && s[s.size()-1] == ']')
    {
        op.type = OperandType::MEMORY;
        op.isAddress = true;
        op.isIndirect = true;  // Memory operands in brackets are indirect
        std::string inner = trim(s.substr(1, s.size() - 2));
        
        // Check for offset notation like [register+offset] or [%r0+4]
        size_t plusPos = inner.find('+');
        if (plusPos != std::string::npos)
        {
            std::string baseReg = trim(inner.substr(0, plusPos));
            std::string offsetStr = trim(inner.substr(plusPos + 1));
            
            // Parse the base register if it starts with %
            if (!baseReg.empty() && baseReg[0] == '%')
            {
                std::string numPart;
                for (size_t i = 1; i < baseReg.size(); ++i)
                {
                    if (std::isdigit(baseReg[i]))
                    {
                        numPart += baseReg[i];
                    }
                }
                if (!numPart.empty())
                {
                    op.registerIndex = std::stoi(numPart);
                }
            }
            
            // Parse the offset
            try {
                op.address = std::stoull(offsetStr);
            } catch (...) {
                op.address = 0;
            }
        }
        else
        {
            // No offset - could be [register] or [param_name]
            // Check if this is a parameter name (not a register)
            if (!inner.empty() && inner[0] != '%')
            {
                // This is a parameter name - try to resolve it
                if (m_currentFunction != nullptr)
                {
                    for (const auto& param : m_currentFunction->parameters)
                    {
                        if (param.name == inner)
                        {
                            op.address = param.offset;
                            op.isIndirect = false;  // Parameter access is not register-indirect
                            return op;
                        }
                    }
                }
                // If not found, default to offset 0
                op.address = 0;
                op.isIndirect = false;
            }
            else if (!inner.empty() && inner[0] == '%')
            {
                // Register indirect addressing like [%r0]
                // Parse the register number
                std::string numPart;
                for (size_t i = 1; i < inner.size(); ++i)
                {
                    if (std::isdigit(inner[i]))
                    {
                        numPart += inner[i];
                    }
                }
                if (!numPart.empty())
                {
                    op.registerIndex = std::stoi(numPart);
                }
                op.address = 0;  // No offset
            }
            else
            {
                // Empty or unknown
                op.address = 0;
            }
        }
        return op;
    }
    if (!s.empty() && s[0] == '%')
    {
        op.type = OperandType::REGISTER;
        std::string numPart;
        for (size_t i = 1; i < s.size(); ++i)
        {
            if (std::isdigit(s[i]))
            {
                numPart += s[i];
            }
        }
        if (!numPart.empty())
        {
            op.registerIndex = std::stoi(numPart);
        }
        return op;
    }
    if (!s.empty() && s[0] == 'p' && s.size() > 1 && std::isdigit(s[1]))
    {
        op.type = OperandType::PREDICATE;
        op.predicateIndex = std::stoi(s.substr(1));
        return op;
    }
    if (!s.empty() && (std::isdigit(s[0]) || s[0] == '-' || s.find('.') != std::string::npos))
    {
        op.type = OperandType::IMMEDIATE;
        try
        {
            op.immediateValue = std::stoll(s);
        }
        catch (...)
        {
            op.immediateValue = 0;
        }
        return op;
    }
    op.type = OperandType::UNKNOWN;
    return op;
}

// PTXParser公共接口实现
PTXParser::PTXParser() : pImpl(std::make_unique<Impl>()) {}
PTXParser::~PTXParser() = default;
bool PTXParser::parseFile(const std::string &filename) { return pImpl->parseFile(filename); }
bool PTXParser::parseString(const std::string &ptxCode) { return pImpl->parseString(ptxCode); }
const std::vector<DecodedInstruction> &PTXParser::getInstructions() const { return pImpl->getInstructions(); }
const std::string &PTXParser::getErrorMessage() const { return pImpl->getErrorMessage(); }
const PTXProgram &PTXParser::getProgram() const { return pImpl->getProgram(); }
