/*  Lattice Boltzmann sample, written in C++, using the OpenLB
 *  library
 *
 *  Copyright (C) 2024 Adrian Kummerlaender
 *  E-mail contact: info@openlb.net
 *  The most recent release of OpenLB can be downloaded at
 *  <http://www.openlb.net/>
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation; either version 2
 *  of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public
 *  License along with this program; if not, write to the Free
 *  Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 *  Boston, MA  02110-1301, USA.
 */
#include <olb.h>
/* 
 * These are all the lammps header files we need 
 * */
#include "lammps/lammps.h"
#include "lammps/input.h"
#include "lammps/output.h"
#include "lammps/neighbor.h"
#include "lammps/neigh_list.h"
#include "lammps/atom.h"
#include "lammps/comm.h"
#include "lammps/error.h"
#include "lammps/fix.h"
#include "lammps/force.h"
#include "lammps/memory.h"
#include "lammps/modify.h"
#include "lammps/compute.h"
#include "lammps/variable.h"
#include "lammps/library.h"

#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>
#include <sstream>
#include <iostream>

using namespace olb;

using T = float;
using DESCRIPTOR = descriptors::D3Q19<
descriptors::POROSITY,
	descriptors::VELOCITY,
	descriptors::FORCE,
	fields::fsi::ELEMENT_TAG,
	fields::fsi::ELEMENT_FORCE,
	fields::fsi::ELEMENT_TORQUE
	>;

	using BulkDynamics = dynamics::Tuple<
	T, DESCRIPTOR,
	momenta::BulkTuple,
	equilibria::SecondOrder,
	collision::BGK,
	forcing::ForcedHLBM
	>;

const T maxPhysT = 1000.0;
const T writeConfigInterval = 0.01;
const T writeStat = 0.01;
const T updateForce = 1e-4;
const T writeRestart = 0.1;

std::string STLFileName = "rotated_cube_0.000282.stl";
std::string molFileName = "cube0.000282.mol";

T diameterSphere = 0.00035;
int N_side;
T unit_x;
T unit_y;
T unit_z;
T lengthX;
T lengthY;
T lengthZ;

int nRigidBodies;

int resolution = 12; 
T shearViscosity = 1e-3;
T charPhysLength = diameterSphere;
T charPhysVelocity = 0.04;
T densityFluid = 1000; 
T kinematicViscosity = shearViscosity / densityFluid; // m^2/s
T tau = 0.55;

T youngsModulus = 5e4;
T poissonRatio = 0.3;
T coefficientOfRestitution = 0.9;
T coefficientStaticFriction = 0.6;
T coefficientStaticFrictionW = 0.3;
T coefficientKineticFriction = 0.3;

T youngsModulusWall = 1e9;
T poissonRatioWall = 0.3;

const T rho = 2500;
const T diameter = diameterSphere;
const T nu = poissonRatio;
const T beta = 0.8766 + 0.163 * nu;
const T SHEAR_MOD = youngsModulus / (2 * (1 + nu));

const T dtc = M_PI * (diameter / (2. * beta)) * sqrt(rho / SHEAR_MOD);
double LAMMPSTimeStep = 0.01 * dtc;

T acceGrav = (1. - densityFluid / rho) * 9.81;

std::shared_ptr<IndicatorF3D<T>> makeDomainI() {
	std::shared_ptr<IndicatorF3D<T>> boxI(new IndicatorCuboid3D<T>({lengthX, lengthY, lengthZ}, {0, 0, 0}));
	return boxI;
}

T effectiveYoungsModulus = 1. / ((1 - poissonRatio * poissonRatio) / youngsModulus + (1 - poissonRatioWall * poissonRatioWall) / youngsModulusWall);

Vector<T,3> computeAverageFluidVelocity(SuperLattice<T,DESCRIPTOR>& sLattice)
{
	T localSum[3] = {T{0}, T{0}, T{0}};
	int localCount = 0;

	for (int iC = 0; iC < sLattice.getLoadBalancer().size(); ++iC) {
		auto& block = sLattice.getBlock(iC);
		block.forCoreSpatialLocations([&](LatticeR<3> loc) {
				auto cell = block.get(loc);
				//T por = cell.template getField<descriptors::POROSITY>();

				T rho{};
				T u[3]{};
				cell.computeRhoU(rho, u);
				localSum[0] += u[0];
				localSum[1] += u[1];
				localSum[2] += u[2];
				localCount++;
				});
	}

	singleton::mpi().reduceAndBcast(localSum[0], MPI_SUM);
	singleton::mpi().reduceAndBcast(localSum[1], MPI_SUM);
	singleton::mpi().reduceAndBcast(localSum[2], MPI_SUM);
	singleton::mpi().reduceAndBcast(localCount, MPI_SUM);

	if (localCount == 0) return Vector<T,3>{T{0}};
	return Vector<T,3>{localSum[0] / localCount,
		localSum[1] / localCount,
		localSum[2] / localCount};
}

std::string scientific_string(double value, int precision = 15) {
	std::ostringstream oss;
	oss << std::scientific << std::setprecision(precision) << value;
	return oss.str();
}

Vector<T,3> computeTotalForce(SuperLattice<T,DESCRIPTOR>& sLattice, T F)
{
	T localSum[3] = {T{0}, T{0}, T{0}};
	int localCount = 0;

	for (int iC = 0; iC < sLattice.getLoadBalancer().size(); ++iC) {
		auto& block = sLattice.getBlock(iC);
		block.forCoreSpatialLocations([&](LatticeR<3> loc) {
				auto cell = block.get(loc);
				T por = cell.template getField<descriptors::POROSITY>();

				if (por >= T{0.99}) {
				auto f = cell.template getField<descriptors::FORCE>();
				localSum[0] += f[0];
				localSum[1] += f[1];
				localSum[2] += f[2];
				localCount++;
				}
				});
	}

	singleton::mpi().reduceAndBcast(localSum[0], MPI_SUM);
	singleton::mpi().reduceAndBcast(localSum[1], MPI_SUM);
	singleton::mpi().reduceAndBcast(localSum[2], MPI_SUM);
	singleton::mpi().reduceAndBcast(localCount, MPI_SUM);

	if (localCount == 0) return Vector<T,3>{T{0}};
	return Vector<T,3>{localSum[0], localSum[1], localSum[2]};
}

void prepareGeometry(const UnitConverter<T,DESCRIPTOR>& converter, SuperGeometry<T,3>& sGeometry)
{
	OstreamManager clout(std::cout, "prepareGeometry");
	clout << "Preparing geometry..." << std::endl;

	T L = converter.getPhysDeltaX();
	auto domainI = makeDomainI();

	sGeometry.rename(0, 1, domainI);

	Vector<T,3> extend_Bottom(lengthX, lengthY, L);
	Vector<T,3> origin_Bottom(0., 0., -L);
	IndicatorCuboid3D<T> bottombox(extend_Bottom, origin_Bottom);

	Vector<T,3> extend_1(L, lengthY, lengthZ);
	Vector<T,3> origin_1(-L, 0., 0.);
	IndicatorCuboid3D<T> xLbox(extend_1, origin_1);

	Vector<T,3> extend_2(lengthX, L, lengthZ);
	Vector<T,3> origin_2(0., -L, 0.);
	IndicatorCuboid3D<T> yLbox(extend_2, origin_2);

	Vector<T,3> extend_Top(lengthX, lengthY, L);
	Vector<T,3> origin_Top(0., 0., lengthZ);
	IndicatorCuboid3D<T> topbox(extend_Top, origin_Top);

	Vector<T,3> extend_3(L, lengthY, lengthZ);
	Vector<T,3> origin_3(lengthX, 0., 0.);
	IndicatorCuboid3D<T> xHbox(extend_3, origin_3);

	Vector<T,3> extend_4(lengthX, L, lengthZ);
	Vector<T,3> origin_4(0., lengthY, 0.);
	IndicatorCuboid3D<T> yHbox(extend_4, origin_4);

	sGeometry.clean(true);
	sGeometry.checkForErrors();
	sGeometry.print();

	clout << "Geometry preparation complete." << std::endl;
}

void prepareLattice(SuperLattice<T,DESCRIPTOR>& sLattice,
		const UnitConverter<T,DESCRIPTOR>& converter,
		SuperGeometry<T,3>& sGeometry)
{
	OstreamManager clout(std::cout, "prepareLattice");
	clout << "Preparing lattice..." << std::endl;

	sLattice.defineDynamics<NoDynamics>(sGeometry, 0);
	sLattice.defineDynamics<BulkDynamics>(sGeometry, 1);

	boundary::set<boundary::BounceBack>(sLattice, sGeometry, 2);
	boundary::set<boundary::BounceBack>(sLattice, sGeometry, 3);

	sLattice.setParameter<descriptors::OMEGA>(converter.getLatticeRelaxationFrequency());

	AnalyticalConst3D<T,T> rhoF(1);
	AnalyticalConst3D<T,T> uF(0,0);
	auto bulkIndicator = sGeometry.getMaterialIndicator({1});
	sLattice.iniEquilibrium(bulkIndicator, rhoF, uF);
	sLattice.defineRhoU(bulkIndicator, rhoF, uF);

	{
		AnalyticalConst3D<T,T> porosityF(1);
		sLattice.defineField<descriptors::POROSITY>(bulkIndicator, porosityF);
	}

	sLattice.initialize();

	clout << "Lattice preparation complete." << std::endl;
}

void getResults(SuperLattice<T,DESCRIPTOR>& sLattice,
		const UnitConverter<T,DESCRIPTOR>& converter,
		SuperGeometry<T,3>& sGeometry,
		util::Timer<T>& timer,
		std::size_t iT)
{
	OstreamManager clout(std::cout, "getResults");

	const int vtkIter  = converter.getLatticeTime(writeConfigInterval);
	const int statIter = converter.getLatticeTime(writeStat);

	if (iT == 0) {
		SuperVTMwriter3D<T> vtmWriter("flow");
		vtmWriter.createMasterFile();
	}

	if (iT % statIter == 0) 
	{
		timer.update(iT);
		timer.printStep();
		sLattice.getStatistics().print(iT, converter.getPhysTime(iT));
		if (std::isnan(sLattice.getStatistics().getAverageRho())) {
			std::exit(-1);
		}
	}

	if (iT % vtkIter == 0) 
	{
		sLattice.executePostProcessors(stage::Evaluation{});
		sLattice.setProcessingContext(ProcessingContext::Evaluation);
		sLattice.scheduleBackgroundOutputVTK([&,iT](auto task) {
				SuperVTMwriter3D<T> vtkWriter("flow");
				SuperLatticePhysVelocity3D velocity(sLattice, converter);
				SuperLatticeDensity3D density(sLattice);
				SuperLatticeField3D<T,DESCRIPTOR,descriptors::POROSITY> porosity(sLattice);
				SuperLatticeField3D<T,DESCRIPTOR,fields::fsi::ELEMENT_TAG> tag(sLattice);
				SuperLatticeField3D<T,DESCRIPTOR,fields::fsi::ELEMENT_FORCE> force(sLattice);
				SuperLatticeField3D<T,DESCRIPTOR,descriptors::FORCE> antigravity(sLattice);
				vtkWriter.addFunctor(velocity);
				vtkWriter.addFunctor(porosity);
				vtkWriter.addFunctor(tag);
				vtkWriter.addFunctor(force);
				vtkWriter.addFunctor(density);
				vtkWriter.addFunctor(antigravity);
				vtkWriter.write(iT);
				});
	}
}

void getGlobalRigidPositionVelocityOrientation(double **localXCM, double *globalXCM, 
		double **localVCM, double *globalVCM, 
		double **localE, double *globalE, 
		double **localOmega, double *globalOmega,
		int totalRigidBodies, int localRigidBodies)
{
	double *localXcmArray = new double[totalRigidBodies*3](); 
	double *localVcmArray = new double[totalRigidBodies*3](); 
	double *localOmegaArray = new double[totalRigidBodies*3](); 
	double *localOrientationArray = new double[totalRigidBodies*9](); 

	for(int localIndex=0; localIndex<localRigidBodies; localIndex++)
	{
		int bodyId = int(localXCM[localIndex][0] - 1);
		localXcmArray[3*bodyId]     = localXCM[localIndex][1];
		localXcmArray[3*bodyId + 1] = localXCM[localIndex][2];
		localXcmArray[3*bodyId + 2] = localXCM[localIndex][3];

		localVcmArray[3*bodyId]     = localVCM[localIndex][1];
		localVcmArray[3*bodyId + 1] = localVCM[localIndex][2];
		localVcmArray[3*bodyId + 2] = localVCM[localIndex][3];

		int orientId = int(localE[localIndex][0] - 1);
		for (int k = 0; k < 9; ++k) {
			localOrientationArray[9*orientId + k] = localE[localIndex][k+1];
		}

		localOmegaArray[3*bodyId]     = localOmega[localIndex][1];
		localOmegaArray[3*bodyId + 1] = localOmega[localIndex][2];
		localOmegaArray[3*bodyId + 2] = localOmega[localIndex][3];
	}

	MPI_Allreduce(localXcmArray, globalXCM, totalRigidBodies*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(localVcmArray, globalVCM, totalRigidBodies*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(localOmegaArray, globalOmega, totalRigidBodies*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(localOrientationArray, globalE, totalRigidBodies*9, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	delete[] localXcmArray;
	delete[] localVcmArray;
	delete[] localOmegaArray;
	delete[] localOrientationArray;
}

void setupLAMMPS(LAMMPS_NS::LAMMPS *lmp, std::shared_ptr<IndicatorF3D<T>> domainI, bool RESTART = false)
{
	OstreamManager clout(std::cout, "setupLAMMPS");
	if(!RESTART)
	{
		lmp->input->one("units si"); 
		lmp->input->one("atom_style hybrid full sphere"); 
		lmp->input->one("atom_modify map yes");
		auto min = domainI->getMin();
		auto max = domainI->getMax();
		std::string regionCommand = "region BOX block " + scientific_string(min[0]) + " " + scientific_string(max[0]) + " " 
			+ scientific_string(min[1]) + " " + scientific_string(max[1]) + " " 
			+ scientific_string(min[2]) + " " + scientific_string(max[2]);
		lmp->input->one(regionCommand);
		lmp->input->one("create_box 1 BOX");
		lmp->input->one("change_box all boundary p p p");
	}
	lmp->input->one("group DUMMY empty");
	lmp->input->one("fix grav DUMMY gravity " + scientific_string(acceGrav) + " vector 0 0 -1");
	lmp->input->one("molecule TETRA " + molFileName);
}

void postParticleInsertionSetupLAMMPS(LAMMPS_NS::LAMMPS *lmp, double cutOFF, std::string filename)
{
	lmp->input->one("pair_style granular");
	lmp->input->one("pair_coeff * * hertz/material " + scientific_string(youngsModulus) + " " + scientific_string(coefficientOfRestitution) + " " + scientific_string(poissonRatio) + " tangential mindlin NULL 1.0 " + scientific_string(coefficientStaticFriction) + " damping coeff_restitution");

	lmp->input->one("comm_modify vel yes");
	lmp->input->one("compute t1 all temp/sphere");
	lmp->input->one("thermo_modify temp t1");
	lmp->input->one("group PFP type 1");
	lmp->input->one("fix 1 PFP rigid/small molecule gravity grav infile " + filename);
	lmp->input->one("neighbor " + scientific_string(cutOFF) + " bin");

	lmp->input->one("neigh_modify exclude molecule/intra PFP");
	lmp->input->one("dump total_force all custom 1 check.dat id mol mass x y z radius vx vy vz");
	lmp->input->one("run 0");
	lmp->input->one("undump total_force");

	lmp->input->one("comm_modify vel yes");
	lmp->input->one("timestep " + scientific_string(LAMMPSTimeStep));

	lmp->input->one("run 0");
	lmp->input->one("neigh_modify check yes");
	lmp->input->one("compute RIGID_BODY_ATTRIBUTES_xcm all rigid/local 1 mol x y z");
	lmp->input->one("compute RIGID_BODY_ATTRIBUTES_vcm all rigid/local 1 mol vx vy vz");
	lmp->input->one("compute RIGID_BODY_ATTRIBUTES_quat all rigid/local 1 mol quatw quati quatj quatk");
	lmp->input->one("compute RIGID_BODY_ATTRIBUTES_inertia all rigid/local 1 mol inertiax inertiay inertiaz");
	lmp->input->one("compute RIGID_BODY_ATTRIBUTES_omega all rigid/local 1 mol omegax omegay omegaz");
	lmp->input->one("compute RIGID_BODY_ATTRIBUTES_ex_space all rigid/local 1 mol exx exy exz eyx eyy eyz ezx ezy ezz");
	lmp->input->one("thermo 1000");
}

Vector<T,DESCRIPTOR::d*(DESCRIPTOR::d+1)> setupReferencePorosity(
		const UnitConverter<T,DESCRIPTOR>& converter,
		std::shared_ptr<BlockD<T,descriptors::fsi::REFERENCE_POROSITY_ELEMENT<DESCRIPTOR::d>>>& cubeD,
		BlockLoadBalancer<T> loadBalancer,
		std::shared_ptr<IndicatorF3D<T>>& cubeBoundingI,
		double rho,
		double& mass,
		double& volume)
{
	OstreamManager clout(std::cout, "setupReferencePorosity");
	Vector xcmMinusObject{0.,0.,0.};
	const T elementDeltaX = converter.getPhysDeltaX() / 2;
	Vector<std::size_t,DESCRIPTOR::d> cubeDim;
	Vector xcm{0.,0.,0.};
	Vector inertiaTensor{0.,0.,0.,0.,0.,0.,0.,0.,0.};

	clout << "Setting up reference porous particle..." << std::endl;
	std::shared_ptr<IndicatorF3D<T>> cubeI(new STLreader<T>(STLFileName, elementDeltaX, 1., RayMode::Robust));
	cubeBoundingI = std::make_shared<IndicatorLayer3D<T>>(*cubeI, converter.getPhysDeltaX());
	cubeDim = (cubeBoundingI->getMax() - cubeBoundingI->getMin()) / elementDeltaX + 1.5;

	cubeD.reset(makeSharedBlockD<T,descriptors::fsi::REFERENCE_POROSITY_ELEMENT<DESCRIPTOR::d>>(loadBalancer, cubeDim).release());
	int nVoxels = 0;
	cubeD->forCoreSpatialLocations([&](LatticeR<DESCRIPTOR::d> latticeR) {
			bool inside{};
			Vector<T,DESCRIPTOR::d> physR = cubeBoundingI->getMin() + latticeR * elementDeltaX;
			cubeI->operator()(&inside, physR.data());
			if (inside) {
			xcm = xcm + physR;
			nVoxels++;
			cubeD->setField<descriptors::POROSITY>(latticeR, 0);
			} else {
			cubeD->setField<descriptors::POROSITY>(latticeR, 1);
			}
			});
	xcm = xcm / nVoxels;

	cubeD->forCoreSpatialLocations([&](LatticeR<DESCRIPTOR::d> latticeR) {
			bool inside{};
			Vector<T,DESCRIPTOR::d> physR = cubeBoundingI->getMin() + latticeR * elementDeltaX;
			cubeI->operator()(&inside, physR.data());
			if (inside) {
			Vector physRminusXcm = physR - xcm;
			inertiaTensor[0] += elementDeltaX*elementDeltaX*elementDeltaX*rho*(physRminusXcm[1]*physRminusXcm[1]+physRminusXcm[2]*physRminusXcm[2]);
			inertiaTensor[4] += elementDeltaX*elementDeltaX*elementDeltaX*rho*(physRminusXcm[0]*physRminusXcm[0]+physRminusXcm[2]*physRminusXcm[2]);
			inertiaTensor[8] += elementDeltaX*elementDeltaX*elementDeltaX*rho*(physRminusXcm[1]*physRminusXcm[1]+physRminusXcm[0]*physRminusXcm[0]);
			inertiaTensor[1] -= elementDeltaX*elementDeltaX*elementDeltaX*rho*(physRminusXcm[0]*physRminusXcm[1]);
			inertiaTensor[2] -= elementDeltaX*elementDeltaX*elementDeltaX*rho*(physRminusXcm[0]*physRminusXcm[2]);
			inertiaTensor[5] -= elementDeltaX*elementDeltaX*elementDeltaX*rho*(physRminusXcm[1]*physRminusXcm[2]);
			}
			});
	cubeD->setProcessingContext(ProcessingContext::Simulation);

	inertiaTensor[3] = inertiaTensor[1];
	inertiaTensor[6] = inertiaTensor[2];
	inertiaTensor[7] = inertiaTensor[5];

	mass = nVoxels * elementDeltaX * elementDeltaX * elementDeltaX * rho;
	volume = nVoxels * elementDeltaX * elementDeltaX * elementDeltaX;

	return Vector{xcm[0], xcm[1], xcm[2],
		inertiaTensor[0], inertiaTensor[1], inertiaTensor[2],
		inertiaTensor[3], inertiaTensor[4], inertiaTensor[5],
		inertiaTensor[6], inertiaTensor[7], inertiaTensor[8]};
}

int main(int argc, char* argv[]) {

	T phi = std::stof(argv[2]);
	nRigidBodies = 3150;

	T boxSize = pow(phi / (nRigidBodies * (4./3.) * M_PI * pow(diameterSphere / 2., 3)), -1./3.);

	lengthX = boxSize;
	lengthY = boxSize;
	lengthZ = boxSize;

	initialize(&argc, &argv);
	singleton::directories().setOutputDir(argv[1]);

	OstreamManager clout(std::cout, "main");
	CLIreader args(argc, argv);

	LAMMPS_NS::LAMMPS *lmp;
	const char *lmpargv[3];
	lmpargv[0] = "liblammps";
	lmpargv[1] = "-screen";
	lmpargv[2] = "out.log";
	int lmpargc = sizeof(lmpargv) / sizeof(const char *);

	lmp = new LAMMPS_NS::LAMMPS(lmpargc, (char **)lmpargv, MPI_COMM_WORLD);
	clout << "[LAMMPS] Version ID: " << lmp->num_ver << std::endl;

	const int N = args.getValueOrFallback("--resolution", resolution);

	const UnitConverterFromResolutionAndRelaxationTime<T,DESCRIPTOR> converter(
			N, (T) tau, (T) charPhysLength, (T) charPhysVelocity, (T) kinematicViscosity, (T) densityFluid
			);

	converter.print();
	bool RESTART = false;

	auto domainI = makeDomainI();

	CuboidDecomposition<T,3> cuboidDecomposition(*domainI, converter.getPhysDeltaX(), singleton::mpi().getSize());
	cuboidDecomposition.setPeriodicity({true, true, true});
	BlockLoadBalancer<T> loadBalancer(cuboidDecomposition);

	SuperGeometry<T,3> sGeometry(cuboidDecomposition, loadBalancer);
	prepareGeometry(converter, sGeometry);

	SuperLattice sLattice(converter, sGeometry);
	prepareLattice(sLattice, converter, sGeometry);

	SuperPorousElementEmbeddingO fsiEmbeddingO(sLattice);
	fsiEmbeddingO.registerElementType<ReferenceLatticePorosityF>(sGeometry.getMaterialIndicator({1,2,3}));

	SuperPorousElementReductionO<T,DESCRIPTOR,fields::fsi::ELEMENT_FORCE,fields::fsi::ELEMENT_TORQUE> fsiReductionO(sLattice, sGeometry.getMaterialIndicator({1,2,3}));
	fsiReductionO.resize(nRigidBodies);
	fsiReductionO.addCollectionO(meta::id<CollectPorousBoundaryForceAndTorqueO>{});

	std::shared_ptr<BlockD<T,descriptors::fsi::REFERENCE_POROSITY_ELEMENT<DESCRIPTOR::d>>> cubeD;
	Vector<std::size_t,DESCRIPTOR::d> cubeDim;
	std::shared_ptr<IndicatorF3D<T>> cubeBoundingI;
	const T elementDeltaX = converter.getPhysDeltaX() / 2;
	double mass; 
	double volume; 
	Vector xcmMinusObject = {0., 0., 0.};
	std::string filename; 

	Vector inertiaTensor = {0., 0., 0., 0., 0., 0., 0., 0., 0.};
	std::ifstream inFile("initial_xyz.dat");

	if(!RESTART)
	{
		filename = "infileForRigidBodies.txt";
		std::ofstream outFile; 

		if (singleton::mpi().getRank() == 0) {
			outFile.open(filename, std::ios::out | std::ios::trunc);
			outFile << nRigidBodies << "\n";
		}

		setupLAMMPS(lmp, domainI);
		auto xcmInertiaTensor = setupReferencePorosity(converter, cubeD, loadBalancer, cubeBoundingI, rho, mass, volume);

		inertiaTensor = {xcmInertiaTensor[3], xcmInertiaTensor[4], xcmInertiaTensor[5],
			xcmInertiaTensor[6], xcmInertiaTensor[7], xcmInertiaTensor[8],
			xcmInertiaTensor[9], xcmInertiaTensor[10], xcmInertiaTensor[11]};
		Vector xcm = {xcmInertiaTensor[0], xcmInertiaTensor[1], xcmInertiaTensor[2]};

		xcmMinusObject = xcm - cubeBoundingI->getMin();
		cubeDim = (cubeBoundingI->getMax() - cubeBoundingI->getMin()) / elementDeltaX + 1.5;

		// Structured simulation summary block
		clout << "=================== Physical System Specs ===================" << std::endl;
		clout << "  Rigid Bodies Count: " << nRigidBodies << std::endl;
		clout << "  Domain Size (m):    " << lengthX << " x " << lengthY << " x " << lengthZ << std::endl;
		clout << "  Particle Mass (kg): " << mass << std::endl;
		clout << "  Particle Vol (m^3): " << volume << std::endl;
		clout << "  Packing Fraction:   " << (nRigidBodies * volume / (lengthX * lengthY * lengthZ)) << std::endl;
		clout << "=============================================================" << std::endl;

		if (fsiReductionO.rankDoesFSI()) 
		{
			clout << "Inserting particles into system..." << std::endl;
			int count = 0;
			for(int i = 0; i < nRigidBodies; i++)
			{
				count++;
				double x_coord, y_coord, z_coord;
				inFile >> x_coord >> y_coord >> z_coord;

				x_coord = x_coord / (16 * diameterSphere) * lengthX;
				y_coord = y_coord / (16 * diameterSphere) * lengthY;
				z_coord = z_coord / (16 * diameterSphere) * lengthZ;

				x_coord = x_coord - lengthX * std::floor(x_coord / lengthX);
				y_coord = y_coord - lengthY * std::floor(y_coord / lengthY);
				z_coord = z_coord - lengthZ * std::floor(z_coord / lengthZ);

				// Progress indicator replacing 3,150 lines of stdout spam
				if (count % 1000 == 0 || count == nRigidBodies) {
					clout << "  [Progress] Inserted " << count << " / " << nRigidBodies << " particles" << std::endl;
				}

				Vector pivot{x_coord, y_coord, z_coord};
				Vector lower = pivot - xcmMinusObject;
				Vector rotation = {0., 0., 1e-3}; 
				auto rotationMatrix = util::calculateRotationMatrix<T,DESCRIPTOR::d>(rotation);

				auto RmI = util::matrixMatrixProduct<T,DESCRIPTOR::d>(rotationMatrix, inertiaTensor);
				auto RmImRt = util::matrixMatrixProduct<T,DESCRIPTOR::d>(RmI, util::invertRotationMatrix<T,DESCRIPTOR::d>(rotationMatrix));
				Vector eigenVector = {rotationMatrix[7]-rotationMatrix[5], rotationMatrix[2]-rotationMatrix[6], rotationMatrix[3]-rotationMatrix[1]};
				T angle = std::acos(((rotationMatrix[0]+rotationMatrix[4]+rotationMatrix[8])-1)/2.);

				auto params = makeParametersD<T,descriptors::SPATIAL_DESCRIPTOR<3>>(
						fields::fsi::ELEMENT_TAG{}, count,
						fields::fsi::ELEMENT_LOWER{}, lower,
						fields::fsi::ELEMENT_PIVOT{}, pivot,
						fields::fsi::ELEMENT_ROTATION_MATRIX{}, rotationMatrix,
						fields::fsi::ELEMENT_REFERENCE_DELTA_X{}, elementDeltaX,
						fields::fsi::ELEMENT_REFERENCE_EXTENT{}, cubeDim,
						fields::fsi::ELEMENT_REFERENCE_POROSITY{}, cubeD->getField<descriptors::POROSITY>(),
						fields::fsi::PERIODIC_DIRECTIONS{}, Vector<T,DESCRIPTOR::d>{1., 1., 1.},
						fields::fsi::DOMAIN_LENGTH{}, Vector<T,DESCRIPTOR::d>{lengthX, lengthY, lengthZ}
						);
				fsiEmbeddingO.add(params);

				std::string addRigidBody = "create_atoms " + std::to_string(int(0)) + " single " + scientific_string(pivot[0]) + " " + scientific_string(pivot[1]) + " " + scientific_string(pivot[2]) + " mol TETRA 12 rotate " + scientific_string(angle*180./M_PI) + " " + scientific_string(eigenVector[0]) + " " + scientific_string(eigenVector[1]) + " " + scientific_string(eigenVector[2]); 

				if(singleton::mpi().getRank() == 0)
				{
					outFile << count << "\t" << mass << "\t" << pivot[0] << "\t" << pivot[1] << "\t" << pivot[2] << "\t" << RmImRt[0] << "\t" << RmImRt[4] << "\t" << RmImRt[8] << "\t" << RmImRt[1] << "\t" << RmImRt[2] << "\t" << RmImRt[5] << "\t" << "0\t0\t0\t0\t0\t0\t0\t0\t0\t" << rotationMatrix[0] << "\t" << rotationMatrix[3] << "\t" << rotationMatrix[6] << "\t" << rotationMatrix[1] << "\t" << rotationMatrix[4] << "\t" << rotationMatrix[7] << "\t" << rotationMatrix[2] << "\t" << rotationMatrix[5] << "\t" << rotationMatrix[8] << "\t" << inertiaTensor[0] << "\t" << inertiaTensor[4] << "\t" << inertiaTensor[8] << "\n";
				}

				lmp->input->one(addRigidBody);
			}
		}
		if (singleton::mpi().getRank() == 0) {
			outFile.close();
		}
	}
	else
	{
		filename = "infileRestart_47957.txt";

		auto xcmInertiaTensor = setupReferencePorosity(converter, cubeD, loadBalancer, cubeBoundingI, rho, mass, volume);
		inertiaTensor = {xcmInertiaTensor[3], xcmInertiaTensor[4], xcmInertiaTensor[5],
			xcmInertiaTensor[6], xcmInertiaTensor[7], xcmInertiaTensor[8],
			xcmInertiaTensor[9], xcmInertiaTensor[10], xcmInertiaTensor[11]};
		Vector xcm = {xcmInertiaTensor[0], xcmInertiaTensor[1], xcmInertiaTensor[2]};
		xcmMinusObject = xcm - cubeBoundingI->getMin();

		cubeDim = (cubeBoundingI->getMax() - cubeBoundingI->getMin()) / elementDeltaX + 1.5;
		setupLAMMPS(lmp, domainI);

		sLattice.load("fluid_checkpoint_discard");
		if (fsiReductionO.rankDoesFSI()) 
		{
			std::ifstream file(filename);
			std::string line;
			std::getline(file, line);

			Vector rotationMatrix = {0.,0.,0.,0.,0.,0.,0.,0.,0.};
			Vector XCM   = {0.,0.,0.};	
			Vector VCM   = {0.,0.,0.};
			Vector OMEGA = {0.,0.,0.};
			while (std::getline(file, line)) {
				if (line.empty()) continue;

				std::stringstream lineStream(line);
				int id;
				double value;

				lineStream >> id;

				int count = 1;
				while (lineStream >> value) {
					if(count > 1 and count < 5) XCM[count-2] = value;
					if(count > 10 and count < 14) VCM[count-11] = value;
					if(count > 13 and count < 17) OMEGA[count-14] = value;
					if(count > 19 and count < 29) rotationMatrix[count-20] = value;
					count++;
				}

				rotationMatrix = util::invertRotationMatrix<T,DESCRIPTOR::d>(rotationMatrix);
				Vector lower = XCM - xcmMinusObject;
				auto params = makeParametersD<T,descriptors::SPATIAL_DESCRIPTOR<3>>(
						fields::fsi::ELEMENT_TAG{}, id,
						fields::fsi::ELEMENT_LOWER{}, lower,
						fields::fsi::ELEMENT_PIVOT{}, XCM,
						fields::fsi::ELEMENT_ROTATION_MATRIX{}, rotationMatrix,
						fields::fsi::ELEMENT_REFERENCE_DELTA_X{}, elementDeltaX,
						fields::fsi::ELEMENT_REFERENCE_EXTENT{}, cubeDim,
						fields::fsi::ELEMENT_REFERENCE_POROSITY{}, cubeD->getField<descriptors::POROSITY>(),
						fields::fsi::ELEMENT_U_TRANSLATION{}, VCM,
						fields::fsi::ELEMENT_U_ROTATION{}, OMEGA,
						fields::fsi::PERIODIC_DIRECTIONS{}, Vector<T,DESCRIPTOR::d>{1., 1., 1.},
						fields::fsi::DOMAIN_LENGTH{}, Vector<T,DESCRIPTOR::d>{lengthX, lengthY, lengthZ}
						);
				fsiEmbeddingO.add(params);
				Vector eigenVector = {rotationMatrix[7]-rotationMatrix[5], rotationMatrix[2]-rotationMatrix[6], rotationMatrix[3]-rotationMatrix[1]};
				T angle;
				if(((rotationMatrix[0]+rotationMatrix[4]+rotationMatrix[8])-1)/2. < -1)
					angle = std::acos(-1.0);
				else if(((rotationMatrix[0]+rotationMatrix[4]+rotationMatrix[8])-1)/2. > 1)
					angle = std::acos(1.0);
				else
					angle = std::acos(((rotationMatrix[0]+rotationMatrix[4]+rotationMatrix[8])-1)/2.);

				std::string addRigidBody = "create_atoms " + std::to_string(int(0)) + " single " + scientific_string(XCM[0]) + " " + scientific_string(XCM[1]) + " " + scientific_string(XCM[2]) + " mol TETRA 12 rotate " + scientific_string(angle*180./M_PI) + " " + scientific_string(eigenVector[0]) + " " + scientific_string(eigenVector[1]) + " " + scientific_string(eigenVector[2]); 

				lmp->input->one(addRigidBody);
			}
		}
	}

	std::vector<T> F(3, T());
	F[2] = nRigidBodies * converter.getLatticeForce(mass * acceGrav) / (converter.getLatticeLength(lengthX) * converter.getLatticeLength(lengthY) * converter.getLatticeLength(lengthZ));
	AnalyticalConst3D<T,T> f(F);

	T forceDensityX = F[0];
	T forceDensityY = F[1];
	T forceDensityZ = F[2];

	sLattice.defineField<descriptors::FORCE>(sGeometry, 1, f);
	clout << "[Fluid Force] Initial force applied to fluid nodes (Z): " << F[2] << std::endl;

	T maxDim = -9999999;
	cubeDim = (cubeBoundingI->getMax() - cubeBoundingI->getMin()) / elementDeltaX + 1.5;
	auto cubeDimPhysUnits = (cubeBoundingI->getMax() - cubeBoundingI->getMin());
	for(int i = 0; i < 3; i++) {
		if(maxDim < cubeDimPhysUnits[i]) maxDim = cubeDimPhysUnits[i];
	}

	MPI_Barrier(MPI_COMM_WORLD);
	postParticleInsertionSetupLAMMPS(lmp, 0.1 * maxDim, filename);

	sLattice.setParameter<fields::converter::PHYS_LENGTH>(1 / converter.getConversionFactorLength());
	sLattice.setParameter<fields::converter::PHYS_VELOCITY>(1 / converter.getConversionFactorVelocity());
	sLattice.setParameter<fields::converter::PHYS_DELTA_X>(converter.getPhysDeltaX());

	sLattice.setProcessingContext(ProcessingContext::Simulation);
	fsiEmbeddingO.initialize();

	auto fix_rigid = lmp->modify->get_fix_by_id("1");
	util::Timer<T> timer(converter.getLatticeTime(maxPhysT), sGeometry.getStatistics().getNvoxel());
	timer.start();

	double *externalForceToLAMMPS = new double [nRigidBodies * 3]();
	double *externalTorqueToLAMMPS = new double [nRigidBodies * 3]();

	fix_rigid->assign_variables(externalForceToLAMMPS, externalTorqueToLAMMPS);

	double *globalRigidXcms = new double [nRigidBodies * 3];
	double *globalRigidVcms = new double [nRigidBodies * 3];
	double *globalOrientation = new double [nRigidBodies * 9];
	double *globalOmega = new double [nRigidBodies * 3];

	double OLBPhysDeltaT = converter.getPhysDeltaT();
	int nTimeStep = int(OLBPhysDeltaT / LAMMPSTimeStep);
	if(nTimeStep == 0) {
		nTimeStep = 1; 
		LAMMPSTimeStep = OLBPhysDeltaT;
	}
	LAMMPSTimeStep = OLBPhysDeltaT / nTimeStep;

	// Structured time-step coupling banner
	clout << "================ Execution & Coupling Params ================" << std::endl;
	clout << "  OpenLB Delta T (s):       " << OLBPhysDeltaT << std::endl;
	clout << "  LAMMPS Delta T (s):       " << LAMMPSTimeStep << std::endl;
	clout << "  LAMMPS Sub-steps per LBM: " << nTimeStep << std::endl;
	clout << "  Config Dump Frequency:    Every " << int(writeConfigInterval / LAMMPSTimeStep) << " sub-steps" << std::endl;
	clout << "  Effective Buoyant Gravity:" << acceGrav << " m/s^2" << std::endl;
	clout << "=============================================================" << std::endl;

	lmp->input->one("timestep " + scientific_string(LAMMPSTimeStep));
	lmp->input->one("dump configs all custom " + std::to_string(int(writeConfigInterval / LAMMPSTimeStep)) + " " + argv[1] + "config*.dat id mol diameter mass x y z vx vy vz radius");

	auto computeXcm = lmp->modify->get_compute_by_id("RIGID_BODY_ATTRIBUTES_xcm");
	auto computeVcm = lmp->modify->get_compute_by_id("RIGID_BODY_ATTRIBUTES_vcm");
	auto computeESpace = lmp->modify->get_compute_by_id("RIGID_BODY_ATTRIBUTES_ex_space");
	auto computeOmega = lmp->modify->get_compute_by_id("RIGID_BODY_ATTRIBUTES_omega");
	lmp->input->one("run 0");

	const std::string velocityIteration = "velocityIteration.txt";
	std::ofstream outFileVelocityIteration(argv[1] + velocityIteration, std::ios::out | std::ios::trunc);
	const std::string forcesAndTorques = "forcesTorquesPerParticle.txt";
	std::ofstream forcesAndTorquesFile(argv[1] + forcesAndTorques, std::ios::out | std::ios::trunc);

	int writeVelocityIter = converter.getLatticeTime(writeConfigInterval);
	T Kp = 1e-2;

	std::ofstream outFileRigidRestart;
	clout << "[Simulation] Starting main calculation loop..." << std::endl;

	for (std::size_t iT = 0; iT <= converter.getLatticeTime(maxPhysT); ++iT) 
	{
		sLattice.collideAndStream();
		getResults(sLattice, converter, sGeometry, timer, iT);
		fsiReductionO.apply();

		if(iT % writeVelocityIter == 0 && singleton::mpi().getRank() == 0)
		{
			std::string infile = "infileRestart_" + std::to_string(iT) + ".txt";
			outFileRigidRestart.open(argv[1] + infile, std::ios::trunc);
			outFileRigidRestart << nRigidBodies << std::endl;
		}

		if (fsiReductionO.rankDoesFSI()) 
		{
			for (unsigned iElement = 1; iElement <= fsiReductionO.getElementCount(); ++iElement) 
			{
				auto iRigidBody = fsiEmbeddingO.getField<fields::fsi::ELEMENT_TAG>(iElement - 1)[0];

				auto f = converter.getPhysForce(fsiReductionO.getField<fields::fsi::ELEMENT_FORCE>(iElement));
				auto t = converter.getPhysTorque(fsiReductionO.getField<fields::fsi::ELEMENT_TORQUE>(iElement));

				externalTorqueToLAMMPS[3*(iRigidBody-1) + 0] = t[0];
				externalTorqueToLAMMPS[3*(iRigidBody-1) + 1] = t[1];
				externalTorqueToLAMMPS[3*(iRigidBody-1) + 2] = t[2];

				externalForceToLAMMPS[3*(iRigidBody-1) + 0] = f[0];
				externalForceToLAMMPS[3*(iRigidBody-1) + 1] = f[1];
				externalForceToLAMMPS[3*(iRigidBody-1) + 2] = f[2];
			}   
		}

		std::string runCommand = "run " + std::to_string(nTimeStep) + " pre no\n";
		lmp->input->one(runCommand);

		computeXcm->compute_local();
		computeVcm->compute_local();
		computeESpace->compute_local();
		computeOmega->compute_local();

		getGlobalRigidPositionVelocityOrientation(
				computeXcm->array_local, globalRigidXcms,
				computeVcm->array_local, globalRigidVcms,
				computeESpace->array_local, globalOrientation,
				computeOmega->array_local, globalOmega,
				nRigidBodies, computeXcm->size_local_rows
				);

		if (fsiReductionO.rankDoesFSI()) 
		{
			for (unsigned iElement = 1; iElement <= fsiReductionO.getElementCount(); ++iElement) 
			{
				auto iRigidBody = fsiEmbeddingO.getField<fields::fsi::ELEMENT_TAG>(iElement - 1)[0];

				Vector XCM{0., 0., 0.};
				Vector VCM{0., 0., 0.};
				Vector OMEGA{0., 0., 0.};

				XCM[0] = globalRigidXcms[3*(iRigidBody-1)];
				XCM[1] = globalRigidXcms[3*(iRigidBody-1) + 1];
				XCM[2] = globalRigidXcms[3*(iRigidBody-1) + 2];

				VCM[0] = globalRigidVcms[3*(iRigidBody-1)];
				VCM[1] = globalRigidVcms[3*(iRigidBody-1) + 1];
				VCM[2] = globalRigidVcms[3*(iRigidBody-1) + 2];

				OMEGA[0] = globalOmega[3*(iRigidBody-1)];
				OMEGA[1] = globalOmega[3*(iRigidBody-1) + 1];
				OMEGA[2] = globalOmega[3*(iRigidBody-1) + 2];

				auto rotationMatrixFromLammps = Vector{
					globalOrientation[9*(iRigidBody-1) + 0], globalOrientation[9*(iRigidBody-1) + 3], globalOrientation[9*(iRigidBody-1) + 6],
					globalOrientation[9*(iRigidBody-1) + 1], globalOrientation[9*(iRigidBody-1) + 4], globalOrientation[9*(iRigidBody-1) + 7],
					globalOrientation[9*(iRigidBody-1) + 2], globalOrientation[9*(iRigidBody-1) + 5], globalOrientation[9*(iRigidBody-1) + 8]
				};

				XCM[0] = XCM[0] - lengthX * std::floor(XCM[0] / lengthX);
				XCM[1] = XCM[1] - lengthY * std::floor(XCM[1] / lengthY);
				XCM[2] = XCM[2] - lengthZ * std::floor(XCM[2] / lengthZ);

				fsiEmbeddingO.setField<fields::fsi::ELEMENT_ROTATION_MATRIX>(iElement - 1, rotationMatrixFromLammps);
				fsiEmbeddingO.setField<fields::fsi::ELEMENT_LOWER>(iElement - 1, XCM - xcmMinusObject);
				fsiEmbeddingO.setField<fields::fsi::ELEMENT_UPPER>(iElement - 1, (XCM - xcmMinusObject) + cubeDim * elementDeltaX);
				fsiEmbeddingO.setField<fields::fsi::ELEMENT_PIVOT>(iElement - 1, XCM);
				fsiEmbeddingO.setField<fields::fsi::ELEMENT_U_TRANSLATION>(iElement - 1, VCM);
				fsiEmbeddingO.setField<fields::fsi::ELEMENT_U_ROTATION>(iElement - 1, OMEGA);

				if(singleton::mpi().getRank() == 0 && iT % writeVelocityIter == 0)
				{
					auto RmI = util::matrixMatrixProduct<T,DESCRIPTOR::d>(rotationMatrixFromLammps, inertiaTensor);
					auto RmImRt = util::matrixMatrixProduct<T,DESCRIPTOR::d>(RmI, util::invertRotationMatrix<T,DESCRIPTOR::d>(rotationMatrixFromLammps));
					auto angularMomentum = util::matrixVectorProduct<T,DESCRIPTOR::d>(inertiaTensor, OMEGA);

					outFileRigidRestart << iElement << "\t" << mass
						<< "\t" << XCM[0] << "\t" << XCM[1] << "\t" << XCM[2]
						<< "\t" << RmImRt[0] << "\t" << RmImRt[4] << "\t" << RmImRt[8] << "\t" << RmImRt[1] << "\t" << RmImRt[2] << "\t" << RmImRt[5]
						<< "\t" << VCM[0] << "\t" << VCM[1] << "\t" << VCM[2]
						<< "\t" << angularMomentum[0] << "\t" << angularMomentum[1] << "\t" << angularMomentum[2]
						<< "\t0\t0\t0\t"
						<< rotationMatrixFromLammps[0] << "\t" << rotationMatrixFromLammps[3] << "\t" << rotationMatrixFromLammps[6] << "\t"
						<< rotationMatrixFromLammps[1] << "\t" << rotationMatrixFromLammps[4] << "\t" << rotationMatrixFromLammps[7] << "\t"
						<< rotationMatrixFromLammps[2] << "\t" << rotationMatrixFromLammps[5] << "\t" << rotationMatrixFromLammps[8] << "\t"
						<< inertiaTensor[0] << "\t" << inertiaTensor[4] << "\t" << inertiaTensor[8] << std::endl;
				}
			}   

			fsiEmbeddingO.setProcessingContext<Array<fields::fsi::ELEMENT_ROTATION_MATRIX>>(ProcessingContext::Simulation);
			fsiEmbeddingO.setProcessingContext<Array<fields::fsi::ELEMENT_LOWER>>(ProcessingContext::Simulation);
			fsiEmbeddingO.setProcessingContext<Array<fields::fsi::ELEMENT_PIVOT>>(ProcessingContext::Simulation);
			fsiEmbeddingO.setProcessingContext<Array<fields::fsi::ELEMENT_UPPER>>(ProcessingContext::Simulation);
			fsiEmbeddingO.setProcessingContext<Array<fields::fsi::ELEMENT_U_TRANSLATION>>(ProcessingContext::Simulation);
			fsiEmbeddingO.setProcessingContext<Array<fields::fsi::ELEMENT_U_ROTATION>>(ProcessingContext::Simulation);
		}

		if(iT % writeVelocityIter == 0 && singleton::mpi().getRank() == 0)
		{
			outFileRigidRestart.close();
		}

		if(iT % converter.getLatticeTime(updateForce) == 0)
		{
			Vector<T,3> avgVel = computeAverageFluidVelocity(sLattice);
			T currentForceDensityX = forceDensityX - Kp * avgVel[0];
			T currentForceDensityY = forceDensityY - Kp * avgVel[1];
			T currentForceDensityZ = forceDensityZ - Kp * avgVel[2];

			std::vector<T> F(3, T());
			F[0] = currentForceDensityX;
			F[1] = currentForceDensityY;
			F[2] = currentForceDensityZ;
			AnalyticalConst3D<T,T> f(F);
			sLattice.defineField<descriptors::FORCE>(sGeometry, 1, f);

			forceDensityX = currentForceDensityX;
			forceDensityY = currentForceDensityY;
			forceDensityZ = currentForceDensityZ;
		}

		fsiEmbeddingO.apply();

		if(iT % converter.getLatticeTime(writeRestart) == 0)
		{
			clout << "[Checkpoint] Updated Force Density: (" << forceDensityX << ", " << forceDensityY << ", " << forceDensityZ << ")" << std::endl;
			clout << "[Checkpoint] Saving fluid checkpoint: " << (sLattice.save("fluid_checkpoint_discard") ? "Success" : "Failed") << std::endl;
		}
	}

	outFileVelocityIteration.close();
	forcesAndTorquesFile.close();

	delete [] globalRigidXcms;
	delete [] globalRigidVcms;
	delete [] globalOrientation;
	delete [] globalOmega;
	delete [] externalForceToLAMMPS;
	delete [] externalTorqueToLAMMPS;

	sLattice.setProcessingContext(ProcessingContext::Evaluation);

	timer.stop();
	timer.printSummary();
	delete lmp;

	MPI_Finalize();
	return 0;
}
