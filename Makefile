CXX := g++
NVCC := nvcc
LINK := $(CXX)
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic
CPPFLAGS := -I.
LDFLAGS  :=
LDLIBS   :=
DEP_FLAGS := -MMD -MP

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/app.exe

SOURCES := $(wildcard *.cpp) \
           $(wildcard libraries/NKS_Tokenizer/*.cpp) \
           $(wildcard libraries/CLM_Compressor/*.cpp) \
           $(wildcard libraries/NKS_LLM/*.cpp)
SOURCES := $(filter-out libraries/CLM_Compressor/main.cpp,$(SOURCES))

ifeq ($(OS),Windows_NT)
CPPFLAGS += -DNOMINMAX
MSVC_HOST_DIR ?= C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64
NVCC_CCBIN ?= -ccbin "$(MSVC_HOST_DIR)"
endif

ifeq ($(USE_CUDA),1)
CPPFLAGS += -DNKS_ENABLE_CUDA
SOURCES := $(filter-out libraries/NKS_LLM/gpu_backend.cpp,$(SOURCES))
CUDA_SOURCES := $(wildcard libraries/NKS_LLM/*.cu)
CUDA_OBJECTS := $(patsubst %.cu,$(BUILD_DIR)/%.cu.o,$(CUDA_SOURCES))

# NVCC handles C++17 compilation and 64-bit object generation matching CUDA
CXX := $(NVCC) $(NVCC_CCBIN)
LINK := $(NVCC) $(NVCC_CCBIN)
CXXFLAGS := -std=c++17 -O3
DEP_FLAGS := -MD
NVCC_FLAGS := -std=c++17 -O3 $(NVCC_CCBIN)
else
NVCC_FLAGS := -std=c++17 -O3 $(NVCC_CCBIN)
endif

OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPS    := $(OBJECTS:.o=.d) $(CUDA_OBJECTS:.o=.d)

ifeq ($(OS),Windows_NT)
RUN_EXE := $(TARGET)
THREAD_FLAGS :=
define MKDIR_P
if not exist "$(1)" mkdir "$(1)"
endef
define RM_RF
if exist "$(1)" rmdir /S /Q "$(1)"
endef
else
RUN_EXE := ./$(TARGET)
THREAD_FLAGS := -pthread
define MKDIR_P
mkdir -p "$(1)"
endef
define RM_RF
rm -rf "$(1)"
endef
endif

.PHONY: all run clean rebuild

all: $(TARGET)

SIGNTOOL ?= "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/signtool.exe"

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	$(LINK) $(THREAD_FLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)
ifeq ($(OS),Windows_NT)
	-@if exist $(SIGNTOOL) $(SIGNTOOL) sign /fd SHA256 /sha1 39A0435CCCEBB3D20351F992D2AFED520BE7D87E /s My $@ >nul 2>&1
endif

$(BUILD_DIR)/%.o: %.cpp
	@$(call MKDIR_P,$(dir $@))
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $(DEP_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.cu.o: %.cu
	@$(call MKDIR_P,$(dir $@))
	$(NVCC) $(CPPFLAGS) $(NVCC_FLAGS) -MD -c $< -o $@

run: $(TARGET)
	$(RUN_EXE)

clean:
	@$(call RM_RF,$(BUILD_DIR))

rebuild: clean all

-include $(DEPS)
