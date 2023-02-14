# C++ compiler
CXX = g++ 

# necessary compiler flags for using ROOT (root.cern.ch) - remove these if you're not using root
ROOTCFLAGS    = $(shell root-config --cflags)
ROOTGLIBS     = $(shell root-config --glibs)

# ROOT shared library flags
GLIBS = $(filter-out -stdlib=libc++ -pthread , $(ROOTGLIBS))

# some compiler flags
CXXFLAGS = -std=c++11
# ROOT flags
CXXFLAGS += -fPIC $(filter-out -stdlib=libc++ -pthread , $(ROOTCFLAGS))

# location of source code
SRCDIR = ./src/

#location of header files
INCLUDEDIR = ./include/

CXXFLAGS += -I$(INCLUDEDIR)

# location of object files (from compiled library files)
OUTOBJ = ./obj/

CC_FILES := $(wildcard src/*.cc)
HH_FILES := $(wildcard include/*.hh)
OBJ_FILES := $(addprefix $(OUTOBJ),$(notdir $(CC_FILES:.cc=.o)))

# targets to make
all: CalculatePi.x GaussRandom.x

# recipe for building CalculatePi.x
CalculatePi.x:  $(SRCDIR)CalculatePi.C $(OBJ_FILES) $(HH_FILES)
	$(CXX) $(CXXFLAGS) -o CalculatePi.x $(OUTOBJ)/*.o $(GLIBS) $ $<
	touch CalculatePi.x

# recipe for building CalculatePi.x
GaussRandom.x:  $(SRCDIR)GaussRandom.C $(OBJ_FILES) $(HH_FILES)
	$(CXX) $(CXXFLAGS) -o GaussRandom.x $(OUTOBJ)/*.o $(GLIBS) $ $<
	touch GaussRandom.x

$(OUTOBJ)%.o: src/%.cc include/%.hh
	$(CXX) $(CXXFLAGS) -c $< -o $@

# clean-up target (make clean)
clean:
	rm -f *.x
	rm -rf *.dSYM
	rm -f $(OUTOBJ)*.o
