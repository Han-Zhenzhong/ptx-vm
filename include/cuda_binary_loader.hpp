#ifndef CUDA_BINARY_LOADER_HPP
#define CUDA_BINARY_LOADER_HPP

#include <string>
#include <vector>
#include <cstdint>

// Forward declarations
class CUDABinaryLoader;

typedef uint32_t SymbolType;
typedef uint64_t Elf64_Addr;
typedef uint64_t Elf64_Off;
typedef uint16_t Elf64_Half;
typedef uint32_t Elf64_Word;
typedef int32_t  Elf64_Sword;
typedef uint64_t Elf64_Xword;
typedef int64_t  Elf64_Sxword;

typedef enum {
    // Section types
    SECTION_TYPE_NULL = 0,
    SECTION_TYPE_PROGBITS = 1,
    SECTION_TYPE_SYMTAB = 2,
    SECTION_TYPE_STRTAB = 3,
    SECTION_TYPE_RELA = 4,
    SECTION_TYPE_HASH = 5,
    SECTION_TYPE_DYNAMIC = 6,
    SECTION_TYPE_NOTE = 7,
    SECTION_TYPE_NOBITS = 8,
    SECTION_TYPE_REL = 9,
    SECTION_TYPE_SHLIB = 10,
    SECTION_TYPE_DYNSYM = 11,
    SECTION_TYPE_KERNEL = 0x70000000,  // NVIDIA-specific section type for kernels
} SectionType;

typedef struct {
    // CUDA binary header
    uint32_t magic;         // Magic number (should be CUDA_BINARY_MAGIC)
    uint32_t version;      // Version (major << 16 | minor)
    uint32_t flags;        // Flags
    uint32_t numSections;  // Number of sections
    uint64_t sectionTableOffset;  // Offset to section table
    uint32_t reserved[8];  // Reserved fields
    uint32_t headerSize;   // Size of this header
} CudaBinaryHeader;

typedef struct {
    // Section header
    SectionType type;       // Section type
    uint64_t offset;        // Section file offset
    uint64_t size;          // Section size in bytes
    uint32_t flags;         // Section flags
    uint32_t nameLength;    // Length of section name
    std::string name;       // Section name
    uint64_t dataOffset;    // Offset to section data (from start of file)
    uint64_t dataSize;      // Size of section data
} SectionHeader;

typedef struct {
    // Kernel information
    std::string name;        // Kernel name
    uint32_t flags;          // Kernel flags
    uint64_t ptxCodeOffset; // Offset to PTX code
    uint64_t ptxCodeSize;   // Size of PTX code
    uint32_t paramCount;    // Number of parameters
    uint32_t numRegisters;  // Number of registers used
    uint64_t sharedMemBytes; // Shared memory usage
    uint64_t localMemBytes; // Local memory usage
    uint32_t barrierCount;  // Number of barriers
    uint32_t smemCount;     // Shared memory count
    std::string ptxCode;    // Actual PTX code
} KernelInfo;

typedef struct {
    // Relocation entry
    uint32_t type;          // Relocation type
    uint32_t symbolIndex;   // Symbol index
    uint64_t offset;        // Offset in section
    uint64_t addend;        // Addend value
} RelocationEntry;

typedef struct {
    // Symbol entry
    std::string name;        // Symbol name
    uint32_t type;          // Symbol type
    uint64_t value;         // Symbol value
    uint64_t size;          // Symbol size
} SymbolEntry;

typedef struct {
    // Binary statistics
    uint32_t numKernels;    // Number of kernels
    uint32_t numSections;   // Number of sections
    uint64_t totalCodeSize; // Total code size
    uint64_t totalSharedMemory; // Total shared memory usage
    uint64_t totalLocalMemory;  // Total local memory usage
} BinaryStats;

// FATBIN parsing support
enum FatbinEntryKind {
    FATBIN_ENTRY_PTX = 1,
    FATBIN_ENTRY_CUBIN = 2,
    FATBIN_ENTRY_UNKNOWN = 0
};

class CudaBinaryLoader {
public:
    // Constructor/destructor
    CudaBinaryLoader();
    ~CudaBinaryLoader();

    // Initialize the loader
    bool initialize();

    // Load CUDA binary (FATBIN, CUBIN, or PTX)
    bool loadBinary(const std::string& filename);

    // Load a CUDA binary file
    bool loadCUDABinary(const std::string& filename);

    // Load FATBIN file
    bool loadFatbin(const std::string& filename);

    // Parse PTX from FATBIN entry
    bool parsePTX(const char* data, size_t size);

    // Read CUDA binary header
    bool readCudaBinaryHeader(std::ifstream& file, CudaBinaryHeader& header);

    // Print information about the CUDA binary
    void printBinaryInfo(const CudaBinaryHeader& header);

    // Read all sections from the binary
    bool readSections(std::ifstream& file, const CudaBinaryHeader& header);

    // Read a section header
    bool readSectionHeader(std::ifstream& file, SectionHeader& sectionHeader);

    // Parse a kernel section
    bool parseKernelSection(std::ifstream& file, const SectionHeader& sectionHeader);

    // Read relocation section
    bool readRelocationSection(std::ifstream& file, const SectionHeader& sectionHeader);

    // Process relocations
    bool processRelocations();

    // Apply a single relocation
    bool applyRelocation(const RelocationEntry& reloc);

    // Get PTX code for a specific kernel
    std::string getKernelPTX(const std::string& kernelName);

    // Get list of available kernels
    std::vector<std::string> getAvailableKernels();

    // Get statistics about loaded binary
    BinaryStats getBinaryStats();

private:
    // Private implementation details
    class Impl;
    
    // Core functionality
    bool readSymbolTable(std::ifstream& file, const SectionHeader& sectionHeader);
    bool readStringTable(std::ifstream& file, const SectionHeader& sectionHeader);
    bool readSpecialSection(std::ifstream& file, const SectionHeader& sectionHeader);
    
    // Data members
    bool m_isLoaded = false;
    std::string m_loadedFilename;
    
    // File sections
    std::vector<SectionHeader> m_sections;
    
    // Kernels
    std::vector<KernelInfo> m_kernels;
    
    // Symbol table
    std::vector<SymbolEntry> m_symbolTable;
    
    // String table
    std::string m_stringTable;
    
    // Relocation table
    std::vector<RelocationEntry> m_relocationTable;
    
    // FATBIN structures
    struct FatbinHeader {
        uint32_t magic;
        uint32_t version;
        uint64_t dataOffset;
        uint64_t dataSize;
        uint32_t numEntries;
    };
    
    struct FatbinEntry {
        uint32_t kind;
        uint64_t offset;
        uint64_t size;
        char name[256];
    };

    // FATBIN parsing helper functions
    bool validateFatbinHeader(const FatbinHeader& header);
    bool processFatbinEntry(const FatbinEntry& entry, std::ifstream& file);
    
    // FATBIN entry kinds
    enum FatbinEntryKind {
        FATBIN_ENTRY_PTX = 1,
        FATBIN_ENTRY_CUBIN = 2,
        FATBIN_ENTRY_UNKNOWN = 0
    };
};

// Factory functions
extern "C" {
    CudaBinaryLoader* createCudaBinaryLoader();
    void destroyCudaBinaryLoader(CudaBinaryLoader* loader);
}

#endif // CUDA_BINARY_LOADER_HPP

// Define constants
#define CUDA_BINARY_MAGIC 0x464C457F  // Same as ELF magic for testing
#define MAX_SECTION_NAME_LENGTH 256