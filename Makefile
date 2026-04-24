CXX := g++
NVCC := nvcc
LINK := $(CXX)
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic
CPPFLAGS := -I.
LDFLAGS  :=
LDLIBS   :=

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/app.exe

SOURCES := $(wildcard *.cpp) \
           $(wildcard libraries/NKS_Tokenizer/*.cpp) \
           $(wildcard libraries/CLM_Compressor/*.cpp) \
           $(wildcard libraries/NKS_LLM/*.cpp)
SOURCES := $(filter-out libraries/CLM_Compressor/main.cpp,$(SOURCES))

ifeq ($(USE_CUDA),1)
LINK := $(NVCC)
CPPFLAGS += -DNKS_ENABLE_CUDA
SOURCES := $(filter-out libraries/NKS_LLM/gpu_backend.cpp,$(SOURCES))
CUDA_SOURCES := $(wildcard libraries/NKS_LLM/*.cu)
CUDA_OBJECTS := $(patsubst %.cu,$(BUILD_DIR)/%.cu.o,$(CUDA_SOURCES))
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

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	$(LINK) $(THREAD_FLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: %.cpp
	@$(call MKDIR_P,$(dir $@))
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.cu.o: %.cu
	@$(call MKDIR_P,$(dir $@))
	$(NVCC) $(CPPFLAGS) -std=c++17 -O3 -MMD -MP -c $< -o $@

run: $(TARGET)
	$(RUN_EXE)

clean:
	@$(call RM_RF,$(BUILD_DIR))

rebuild: clean all

-include $(DEPS)
