# Makefile

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -w -O3 -std=c++17 -fPIC -fopenmp

# Source files
SRCS = source/binoctree.cpp source/coarse_step.cpp \
       source/medium_step.cpp source/fine_step.cpp source/dual_contouring.cpp \
       source/bisection.cpp source/event_registry.cpp source/source_splice.cpp \
       source/slicing.cpp

# Object directory
OBJDIR = lib

# Object files (automatically create the corresponding object file names in OBJDIR)
OBJS = $(patsubst source/%.cpp,$(OBJDIR)/%.o,$(SRCS))

# Shared library name
LIB = lib/core.so

# Focused warning gate for the observer/cache/source-splice translation units.
# The slicing exemptions are legacy debt outside this patch's scope; new event
# registry and source-splice code have no warning exemptions.
STRICT_OBJDIR = lib/strict
STRICT_FLAGS = -O2 -std=c++17 -fPIC -fopenmp -Wall -Wextra -Wpedantic -Werror
STRICT_SLICING_EXEMPTIONS = -Wno-error=sign-compare \
	-Wno-error=unused-variable -Wno-error=unused-but-set-variable \
	-Wno-error=return-type

# Default target
all: $(OBJDIR) $(LIB)

strict-focused: $(STRICT_OBJDIR)/event_registry.o \
                $(STRICT_OBJDIR)/source_splice.o \
                $(STRICT_OBJDIR)/slicing.o

$(STRICT_OBJDIR):
	mkdir -p $(STRICT_OBJDIR)

$(STRICT_OBJDIR)/event_registry.o: source/event_registry.cpp | $(STRICT_OBJDIR)
	$(CXX) $(STRICT_FLAGS) -c $< -o $@

$(STRICT_OBJDIR)/source_splice.o: source/source_splice.cpp | $(STRICT_OBJDIR)
	$(CXX) $(STRICT_FLAGS) -c $< -o $@

$(STRICT_OBJDIR)/slicing.o: source/slicing.cpp | $(STRICT_OBJDIR)
	$(CXX) $(STRICT_FLAGS) $(STRICT_SLICING_EXEMPTIONS) -c $< -o $@

# Create the object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Linking step to create the shared library
$(LIB): $(OBJS)
	$(CXX) -shared -o $(LIB) $(OBJS) -fopenmp

# Compilation step
$(OBJDIR)/%.o: source/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(LIB)
	rm -rf $(STRICT_OBJDIR)
