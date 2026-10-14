EXAMPLE = periodicCubes

OLB_ROOT := ../../../external/OpenLB/
include $(OLB_ROOT)/default.mk
CXXFLAGS += -DLAMMPS_SMALLBIG -I../../../external/LAMMPS/build/includes 
LDFLAGS  += -L../../../external/LAMMPS/build -llammps
