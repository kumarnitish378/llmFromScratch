# ============================================================
#  Makefile — llmFromScratch
# ============================================================

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -pthread -Wall -Wextra

TARGET   = llm

# ── Source files ─────────────────────────────────────────────
SRCS  = main.cpp

# Your tokenizer
# SRCS += libraries/NKS_Tokenizer/NKS_Tokenizer.cpp
# SRCS += libraries/NKS_Tokenizer/NKS_SentencePieceTokenizer.cpp

# CLM Compressor (add these 3 lines)
SRCS += libraries/CLM_Compressor/lz77.cpp
SRCS += libraries/CLM_Compressor/huffman.cpp
SRCS += libraries/CLM_Compressor/compressor.cpp

OBJS = $(SRCS:.cpp=.o)

# ── Build rules ───────────────────────────────────────────────
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	del /Q $(subst /,\,$(OBJS)) $(TARGET).exe 2>nul || \
	rm -f $(OBJS) $(TARGET)

rebuild: clean all

.PHONY: all clean rebuild
