# ======================================================================
#  Makefile -- ML_coupling_with_HPC harness
#
#  Targets:
#    all (default)   Build the mlhpc executable.
#    test            Build all tests and run them.
#    clean           Remove object and dependency files.
#    distclean       Remove the entire build/ tree.
#    help            Show this list.
#
#  Variables (override on the command line, e.g. `make CASE=B`):
#    CASE            Coupling case  : A | B   (default A).
#    CXX             C++ compiler   : mpicxx by default; must be an
#                                     MPI wrapper or carry MPI flags
#                                     itself.
#    OPT             Optimisation   : -O2 -g by default.
#    ENABLE_OPENMP   1 to use -fopenmp (default), 0 to disable.
#    ID_32BIT        1 to make IDType = uint32_t.  Default (0) leaves
#                    IDType at uint64_t (production-run-friendly).
#    FEATURE_DOUBLE  1 to make mlhpc::Real = double for the in-memory
#                    cooling-feature dtype.  Default (0) keeps Real
#                    at float (matches single-precision OpenGADGET3
#                    snapshots).
#
#  Build artefacts live in build/  :
#    build/obj/...   object files mirroring the source tree
#    build/dep/...   auto-generated header dependency files
#    build/mlhpc                     the main executable
#    build/test_header_layout        unit test binary
#    build/test_io_gadget2           unit test binary
# ======================================================================

# ---------- user-overridable variables --------------------------------
# Note on CXX: we use ':=' rather than '?=' so the makefile's choice
# wins over any inherited environment $CXX (a common nuisance, since
# many systems pre-set CXX=g++ which lacks the MPI include path).
# Command-line override still works: `make CXX=mpicxx-other`.
CXX            := mpicxx
CASE           ?= A
OPT            ?= -O2 -g
ENABLE_OPENMP  ?= 1
ID_32BIT       ?= 0
FEATURE_DOUBLE ?= 0

# ---------- CASE validation -------------------------------------------
ifeq ($(filter $(CASE),A B),)
  $(error CASE must be A or B (got '$(CASE)'))
endif

# ---------- flags ------------------------------------------------------
WARN_FLAGS    := -Wall -Wextra -Wpedantic \
                 -Wcast-align -Wshadow -Wold-style-cast \
                 -Wnon-virtual-dtor -Woverloaded-virtual \
                 -Wnull-dereference -Wdouble-promotion

INC_FLAGS     := -Iinclude
# OMPI_SKIP_MPICXX / MPICH_SKIP_MPICXX: opt out of the deprecated C++
# MPI bindings (mpi.h would otherwise pull mpicxx.h, which trips our
# strict warning flags with thousands of old-style-cast diagnostics
# inside the MPI library headers themselves).
DEFS          := -DMLHPC_CASE_$(CASE)=1 \
                 -DMLHPC_BACKEND_ONNX=0 \
                 -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1
ifeq ($(ID_32BIT),1)
  DEFS        += -DMLHPC_ID_32BIT=1
endif
ifeq ($(FEATURE_DOUBLE),1)
  DEFS        += -DMLHPC_FEATURE_DOUBLE=1
endif

OPENMP_FLAGS  :=
ifeq ($(ENABLE_OPENMP),1)
  OPENMP_FLAGS := -fopenmp
endif

CXXFLAGS      := -std=c++20 $(OPT) $(WARN_FLAGS) $(OPENMP_FLAGS) \
                 $(INC_FLAGS) $(DEFS)
LDFLAGS       := $(OPENMP_FLAGS)

# ---------- directory layout ------------------------------------------
BUILDDIR      := build
OBJDIR        := $(BUILDDIR)/obj
DEPDIR        := $(BUILDDIR)/dep
BINDIR        := $(BUILDDIR)

# ---------- sources ----------------------------------------------------
# The single-file Format-2 reader, the partition primitive, the
# rank-to-sub-file group distribution, and the MPI-aware Format-2
# distributor live in the support library; main.cpp is the harness
# entry point; everything under tests/ becomes its own binary.
# Exactly one of src/coupling_{A,B}.cpp is compiled, selected by
# the CASE variable.  This keeps make_coupling() / compiled_case_label()
# single-definition at link time and prevents accidentally pulling
# inactive case logic into a build.
CASE_SRC      := src/coupling_$(CASE).cpp

LIB_SRCS      := src/io_gadget2.cpp \
                 src/partition.cpp \
                 src/group_distribution.cpp \
                 src/format2_distributor.cpp \
                 src/units.cpp \
                 src/inference_dummy.cpp \
                 src/inference_factory.cpp \
                 src/coupling_async_mpi.cpp \
                 src/coupling_service.cpp \
                 $(CASE_SRC)
APP_SRCS      := src/main.cpp
# Tests are split into:
#   - non-MPI tests run as standalone executables under `make test`
#   - MPI tests run via `mpirun -np 4` under the same target
TEST_SRCS     := tests/test_header_layout.cpp \
                 tests/test_io_gadget2.cpp \
                 tests/test_partition.cpp \
                 tests/test_group_distribution.cpp \
                 tests/test_units.cpp \
                 tests/test_inference_dummy.cpp
MPI_TEST_SRCS := tests/test_format2_distributor.cpp \
                 tests/test_coupling_A.cpp \
                 tests/test_coupling_B.cpp

ALL_SRCS      := $(LIB_SRCS) $(APP_SRCS) $(TEST_SRCS)

LIB_OBJS      := $(LIB_SRCS:%.cpp=$(OBJDIR)/%.o)
APP_OBJS      := $(APP_SRCS:%.cpp=$(OBJDIR)/%.o)
TEST_OBJS     := $(TEST_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TEST_OBJS := $(MPI_TEST_SRCS:%.cpp=$(OBJDIR)/%.o)
ALL_OBJS      := $(LIB_OBJS) $(APP_OBJS) $(TEST_OBJS) $(MPI_TEST_OBJS)

# ---------- targets ----------------------------------------------------
APP_BIN       := $(BINDIR)/mlhpc
TEST_BINS     := $(BINDIR)/test_header_layout \
                 $(BINDIR)/test_io_gadget2 \
                 $(BINDIR)/test_partition \
                 $(BINDIR)/test_group_distribution \
                 $(BINDIR)/test_units \
                 $(BINDIR)/test_inference_dummy
MPI_TEST_BINS := $(BINDIR)/test_format2_distributor \
                 $(BINDIR)/test_coupling_A \
                 $(BINDIR)/test_coupling_B
MPI_TEST_NP   ?= 4
MPIRUN        ?= mpirun --oversubscribe

.PHONY: all test clean distclean help

all: $(APP_BIN)

# Build every test and run each one; bail out on the first failure.
# Non-MPI tests run standalone; MPI tests run under mpirun.
test: $(TEST_BINS) $(MPI_TEST_BINS)
	@set -e; \
	for t in $(TEST_BINS); do \
	  echo "RUN  $$t"; \
	  $$t; \
	done; \
	for t in $(MPI_TEST_BINS); do \
	  echo "RUN  $(MPIRUN) -np $(MPI_TEST_NP) $$t"; \
	  $(MPIRUN) -np $(MPI_TEST_NP) $$t; \
	done; \
	echo "All tests passed."

# ---------- compilation pattern rule ----------------------------------
# Compile %.cpp -> $(OBJDIR)/%.o, dropping a header-dependency file at
# $(DEPDIR)/%.d via -MMD -MP for incremental rebuilds on header edits.
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D) $(DEPDIR)/$(*D)
	$(CXX) $(CXXFLAGS) -MMD -MP -MF $(DEPDIR)/$*.d -c $< -o $@

# ---------- link rules ------------------------------------------------
$(APP_BIN): $(APP_OBJS) $(LIB_OBJS)
	@mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) -o $@

# test_header_layout is a pure layout check; needs no library object.
$(BINDIR)/test_header_layout: $(OBJDIR)/tests/test_header_layout.o
	@mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) -o $@

# test_io_gadget2 exercises the actual reader, so link the library.
$(BINDIR)/test_io_gadget2: $(OBJDIR)/tests/test_io_gadget2.o $(LIB_OBJS)
	@mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) -o $@

# test_partition is pure arithmetic but still pulls partition.o.
$(BINDIR)/test_partition: $(OBJDIR)/tests/test_partition.o $(OBJDIR)/src/partition.o
	@mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) -o $@

# test_group_distribution exercises the rank-to-sub-file grouping.
$(BINDIR)/test_group_distribution: $(OBJDIR)/tests/test_group_distribution.o \
                                   $(OBJDIR)/src/group_distribution.o \
                                   $(OBJDIR)/src/partition.o
	@mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) -o $@

# test_units validates code -> CGS conversions against hand-computed values.
$(BINDIR)/test_units: $(OBJDIR)/tests/test_units.o $(OBJDIR)/src/units.o
	@mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) -o $@

# test_inference_dummy verifies the deterministic dummy backend.
$(BINDIR)/test_inference_dummy: $(OBJDIR)/tests/test_inference_dummy.o \
                                $(OBJDIR)/src/inference_dummy.o
	@mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) -o $@

# test_format2_distributor exercises the full MPI scatter path.
# Must be launched under mpirun -- the `test` target handles this.
$(BINDIR)/test_format2_distributor: $(OBJDIR)/tests/test_format2_distributor.o $(LIB_OBJS)
	@mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) -o $@

# test_coupling_A exercises the case-A Coupling implementation.
# Built against the same case-specific TU as the main binary, so it
# only verifies meaningful behaviour when CASE=A; under CASE=B it
# skips with an OK line.  Runs under mpirun.
$(BINDIR)/test_coupling_A: $(OBJDIR)/tests/test_coupling_A.o $(LIB_OBJS)
	@mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) -o $@

# test_coupling_B exercises the case-B Coupling implementation.
# Same shape as test_coupling_A: self-skipping under CASE=A|C.  Runs
# under mpirun, default -np 4 (so np=4, pool=1 -> 3 physics + 1
# inference, which is the smallest non-trivial case-B config).
$(BINDIR)/test_coupling_B: $(OBJDIR)/tests/test_coupling_B.o $(LIB_OBJS)
	@mkdir -p $(@D)
	$(CXX) $^ $(LDFLAGS) -o $@

# ---------- housekeeping ----------------------------------------------
clean:
	rm -rf $(OBJDIR) $(DEPDIR)

distclean:
	rm -rf $(BUILDDIR)

help:
	@echo "Targets:"
	@echo "  all          Build the mlhpc binary (default)."
	@echo "  test         Build and run all tests."
	@echo "  clean        Remove object and dependency files."
	@echo "  distclean    Remove the entire build/ tree."
	@echo "Variables (override on the command line):"
	@echo "  CASE          A | B         (default: A)  coupling case."
	@echo "  CXX           compiler     (default: mpicxx)."
	@echo "  OPT           opt flags    (default: -O2 -g)."
	@echo "  ENABLE_OPENMP 1 | 0        (default: 1)."
	@echo "  ID_32BIT      1 | 0        (default: 0; default IDType is uint64_t)."
	@echo "  FEATURE_DOUBLE 1 | 0       (default: 0; default mlhpc::Real is float)."

# ---------- header-dependency includes -------------------------------
# After the first build each .d file pulls in the headers that the
# corresponding .o depends on, so editing a header triggers exactly
# the right recompiles.  Missing on first build is fine (the dash).
-include $(ALL_OBJS:$(OBJDIR)/%.o=$(DEPDIR)/%.d)
