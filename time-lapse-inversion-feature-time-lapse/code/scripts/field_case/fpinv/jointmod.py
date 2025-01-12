import numpy as np
import pygimli as pg

class JointMod(pg.core.ModellingBase):
    def __init__(self, mesh, ertfop, rstfop, petromodel, ntimes=1, fix_poro=True,
                 zWeight=1, verbose=True, corr_l=None, fix_water=False,
                 fix_ice=False, fix_air=False):
        """Joint petrophysical modelling operator.

        Parameters
        ----------
        mesh : pyGIMLi mesh
        ertfop : ERT forward operator
        rstfop : RST forward operator
        petromodel : Petrophysical four-phase model
        zWeight : zWeight for more or less layering
        verbose : Be more verbose
        corr_l : tuple
            Horizontal and vertical correlation lengths. If provided,
            geostatistical regularization will be used and classical smoothing
            with zWeight will be ignored.
        fix_poro|water|ice|air : boolean or vector
            Fix to starting model or provide weight vector for particular cells.
        ntimes : number of time steps (default is one)
        """
        pg.core.ModellingBase.__init__(self, verbose)
        self.mesh = pg.Mesh(mesh)
        self.ERT = ertfop
        self.RST = rstfop
        self.fops = [self.RST, self.ERT]
        self.fpm = petromodel
        self.cellCount = self.mesh.cellCount()
        self.fix_water = fix_water
        self.fix_ice = fix_ice
        self.fix_air = fix_air
        self.fix_poro = fix_poro
        self.zWeight = zWeight
        self.corr_l = corr_l
        self.ntimes = ntimes
        self.createConstraints()


    def fractions(self, model):
        """Split model vector into individual distributions"""

        # Check if multiple time steps (has to be individually calculated here!)
        ntimes = int(len(model)/(4*self.cellCount))

        return np.reshape(model, (4, self.cellCount * ntimes))


    def createJacobian(self, model):

        fw, fi, fa, fr = self.fractions(model)

        # Set number of time steps
        ntimes = self.ntimes

        jacERT = pg.matrix.BlockMatrix()
        jacRST = pg.matrix.BlockMatrix()
        nDataRST, nDataERT = 0, 0

        for timestep in range(1,ntimes+1):

            # Only address fractions for current time step
            idx_t = np.arange((timestep-1)*self.cellCount, timestep*self.cellCount)

            rho = self.fpm.rho(fw[idx_t], fi[idx_t], fa[idx_t], fr[idx_t])
            s = self.fpm.slowness(fw[idx_t], fi[idx_t], fa[idx_t], fr[idx_t])

            self.ERT.fop.createJacobian(rho)
            self.RST.fop.createJacobian(s)

            jacERT_t = self.ERT.fop.jacobian()
            jacRST_t = self.RST.fop.jacobian()

            # Add to block matrix
            idxERT = jacERT.addMatrix(jacERT_t)
            idxRST = jacRST.addMatrix(jacRST_t)

            jacERT.addMatrixEntry(idxERT, nDataERT, self.cellCount * (timestep-1))
            jacRST.addMatrixEntry(idxRST, nDataRST, self.cellCount * (timestep-1))
            nDataRST += self.RST.fop.data.size()
            nDataERT += self.ERT.fop.data.size()

        # Setting inner derivatives
        self.jacRSTW = pg.matrix.MultRightMatrix(jacRST, r=1. / self.fpm.vw)
        self.jacRSTI = pg.matrix.MultRightMatrix(jacRST, r=1. / self.fpm.vi)
        self.jacRSTA = pg.matrix.MultRightMatrix(jacRST, r=1. / self.fpm.va)
        self.jacRSTR = pg.matrix.MultRightMatrix(jacRST, r=1. / self.fpm.vr)

        self.jacERTW = pg.matrix.MultRightMatrix(
            jacERT, r=self.fpm.rho_deriv_fw(fw, fi, fa, fr))
        self.jacERTI = pg.matrix.MultRightMatrix(
            jacERT, r=self.fpm.rho_deriv_fi(fw, fi, fa, fr))
        self.jacERTA = pg.matrix.MultRightMatrix(
            jacERT, r=self.fpm.rho_deriv_fa(fw, fi, fa, fr))
        self.jacERTR = pg.matrix.MultRightMatrix(
            jacERT, r=self.fpm.rho_deriv_fr(fw, fi, fa, fr))

        # Putting subjacobians together in block matrix
        self.jac = pg.matrix.BlockMatrix()
        nData = 0

        jacsRST = [self.jacRSTW, self.jacRSTI, self.jacRSTA, self.jacRSTR]
        jacsERT = [self.jacERTW, self.jacERTI, self.jacERTA, self.jacERTR]

        for jaclist in jacsRST, jacsERT:
            for i in range(4):
                idx = self.jac.addMatrix(jaclist[i])
                self.jac.addMatrixEntry(idx, nData, self.cellCount * i * ntimes)
            nData += self.RST.fop.data.size() * ntimes # Update total vector length
        self.setJacobian(self.jac)


    # With temporal smoothing and Active Time Constraints
    def createConstraints(self, delta=0, eps=0, atc=False):
        # First order smoothness matrix
        self._Ctmp = pg.matrix.SparseMapMatrix()

        if self.corr_l is None:
            pg.info("Using smoothing with zWeight = %.2f." % self.zWeight)
            rm = self.RST.fop.regionManager()
            rm.fillConstraints(self._Ctmp)
            # Set zWeight
            rm.setZWeight(self.zWeight)
            self.cWeight = pg.Vector()
            rm.fillConstraintWeights(self.cWeight)
            self._CW = pg.matrix.LMultRMatrix(self._Ctmp, self.cWeight)
        else:
            pg.info("Using geostatistical constraints with " + str(self.corr_l))
            # Geostatistical constraints by Jordi et al., GJI, 2018
            CM = pg.utils.geostatistics.covarianceMatrix(self.mesh, I=self.corr_l)
            self._Ctmp = pg.matrix.Cm05Matrix(CM)
            self._CW = self._Ctmp

        # Set number of time steps
        ntimes = self.ntimes

        # Putting together in block matrix
        self._C = pg.matrix.BlockMatrix()
        cid = self._C.addMatrix(self._CW)
        for timestep in range(1,ntimes+1):
            self._C.addMatrixEntry(cid, 0 + (timestep-1) * self._Ctmp.rows(), 
                                    0 + (timestep-1) * self.cellCount)
            self._C.addMatrixEntry(cid, self._Ctmp.rows() * (ntimes + timestep-1), 
                                    self.cellCount * (ntimes + timestep-1))
            self._C.addMatrixEntry(cid, self._Ctmp.rows() * (2 * ntimes + timestep-1), 
                                    self.cellCount * (2 * ntimes + timestep-1))
            self._C.addMatrixEntry(cid, self._Ctmp.rows() * (3 * ntimes + timestep-1), 
                                    self.cellCount * (3 * ntimes + timestep-1))

        self._IT = pg.matrix.IdentityMatrix(self.cellCount)
        cid_t = self._C.addMatrix(self._IT)

        if atc == True:
            # Load predefined ATC scaling factors
            deltas = np.load("/home/jokla/notebooks/four-phase-inversion/code/scripts/field_case_timelapse_SCH/deltas.npy")
            D = pg.matrix.DiagonalMatrix(deltas)
            self._ITATC = pg.matrix.Mult2Matrix(
                        pg.matrix.IdentityMatrix(self.cellCount),D)
            cid_tatc = self._C.addMatrix(self._ITATC)
        else:
            cid_tatc = cid_t

        if ntimes == 1:
            # No temporal regularization for one time step
            delta = 0
            epsilon = 0
        else:
            # Scaling temporal regularization strength
            delta = delta * ntimes / (ntimes-1) # Regularization parameter rock content
            epsilon = eps * ntimes / (ntimes-1) # Regularization parameter other phases
        
        for timestep in range(1,ntimes):
            # Water
            self._C.addMatrixEntry(cid_tatc, 
                    self._Ctmp.rows() * 4 * ntimes + self.cellCount * (timestep-1), 
                    self.cellCount * (0 * ntimes + timestep -1), epsilon)
            self._C.addMatrixEntry(cid_tatc, 
                    self._Ctmp.rows() * 4 * ntimes + self.cellCount * (timestep-1), 
                    self.cellCount * (0 * ntimes + timestep), -epsilon)            
            # Ice
            self._C.addMatrixEntry(cid_tatc, 
                    self._Ctmp.rows() * 4 * ntimes + self.cellCount * timestep, 
                    self.cellCount * (1 * ntimes + timestep -1), epsilon)
            self._C.addMatrixEntry(cid_tatc, 
                    self._Ctmp.rows() * 4 * ntimes + self.cellCount * timestep, 
                    self.cellCount * (1 * ntimes + timestep), -epsilon)            
            # Air
            self._C.addMatrixEntry(cid_tatc, 
                    self._Ctmp.rows() * 4 * ntimes + 4 * self.cellCount * (timestep+1), 
                    self.cellCount * (2 * ntimes + timestep -1), epsilon)
            self._C.addMatrixEntry(cid_tatc, 
                    self._Ctmp.rows() * 4 * ntimes + 4 * self.cellCount * (timestep+1), 
                    self.cellCount * (2 * ntimes + timestep), -epsilon)
            # Rock
            self._C.addMatrixEntry(cid_t, 
                    self._Ctmp.rows() * 4 * ntimes + 4 * self.cellCount * (timestep+2), 
                    self.cellCount * (3 * ntimes + timestep -1), delta)
            self._C.addMatrixEntry(cid_t, 
                    self._Ctmp.rows() * 4 * ntimes + 4 * self.cellCount * (timestep+2), 
                    self.cellCount * (3 * ntimes + timestep), -delta)

        self.setConstraints(self._C)

        # Identity matrix for interparameter regularization
        self._I = pg.matrix.IdentityMatrix(self.cellCount * ntimes)

        self._G = pg.matrix.BlockMatrix()
        iid = self._G.addMatrix(self._I)
        self._G.addMatrixEntry(iid, 0, 0)
        self._G.addMatrixEntry(iid, 0, self.cellCount * ntimes)
        self._G.addMatrixEntry(iid, 0, self.cellCount * ntimes * 2)
        self._G.addMatrixEntry(iid, 0, self.cellCount * ntimes * 3)

        self.fix_val_matrices = {}
        # Optionally fix phases to starting model globally or in selected cells
        phases = ["water", "ice", "air", "rock matrix"]
        for i, phase in enumerate([self.fix_water, self.fix_ice, self.fix_air,
                                   self.fix_poro]):
            name = phases[i]
            vec = pg.Vector(self.cellCount)
            if phase is True:
                pg.info("Fixing %s content globally." % name)
                vec += 1.0
            elif hasattr(phase, "__len__"):
                pg.info("Fixing %s content at selected cells." % name)
                phase = np.asarray(phase, dtype="int")
                vec[phase] = 1.0
            self.fix_val_matrices[name] = pg.matrix.DiagonalMatrix(vec)
            mat = self._G.addMatrix(self.fix_val_matrices[name])
            for timestep in range(1,ntimes+1):
                self._G.addMatrixEntry(mat, self._G.rows(), 
                        self.cellCount * (i * ntimes + timestep-1))


    # Chi2 and relative rms for the rhoa data
    def ERTchi2(self, model, error, data):  
        resp = self.response(model)
        resprhoa = resp[self.ntimes * self.RST.fop.data.size():]
        rhoaerr = error[self.ntimes * self.RST.fop.data.size():]
        chi2rhoa = pg.utils.chi2(data, resprhoa, rhoaerr)
        rmsrhoa = pg.utils.rrms(data, resprhoa)
        return chi2rhoa, rmsrhoa

    # Chi2 and relative rms for the travel time data
    def RSTchi2(self, model, error, data):  
        resp = self.response(model)
        resptt = resp[:self.ntimes * self.RST.fop.data.size()]
        tterr = error[:self.ntimes * self.RST.fop.data.size()]
        chi2tt = pg.utils.chi2(data, resptt, tterr)
        rmstt = np.sqrt(np.mean((resptt - data)**2))
        return chi2tt, rmstt


    def response(self, model):
        print(self.RST.fop.mesh())
        print(self.ERT.fop.mesh())
        return self.response_mt(model)


    def response_mt(self, model):
      
        # Set number of time steps
        ntimes = self.ntimes
        print("Number of time steps:",ntimes)
        if ntimes > 1:
            print("More than one time step. Run Time-Lapse Inversion.")

        # Initialize data vectors
        t = []
        rhoa = []
        
        # Loop over time steps
        for timestep in range(1,ntimes+1):
            
            # Forward calculation for one time step
            print("Forward calculation for time step:",timestep)
        
            fw, fi, fa, fr = self.fractions(model)

            # Only address fractions for current time step
            idx_t = np.arange((timestep-1)*self.cellCount, timestep*self.cellCount)

            rho = self.fpm.rho(fw[idx_t], fi[idx_t], fa[idx_t], fr[idx_t])
            s = self.fpm.slowness(fw[idx_t], fi[idx_t], fa[idx_t], fr[idx_t])

            print("=" * 30)
            print("        Min. | Max.")
            print("-" * 30)
            print(" Water: %.2f | %.2f" % (np.min(fw[idx_t]), np.max(fw[idx_t])))
            print(" Ice:   %.2f | %.2f" % (np.min(fi[idx_t]), np.max(fi[idx_t])))
            print(" Air:   %.2f | %.2f" % (np.min(fa[idx_t]), np.max(fa[idx_t])))
            print(" Rock:  %.2f | %.2f" % (np.min(fr[idx_t]), np.max(fr[idx_t])))
            print("-" * 30)
            print(" SUM:   %.2f | %.2f" % (np.min(fa[idx_t] + fw[idx_t]
                                                    + fi[idx_t] + fr[idx_t]),
                                           np.max(fa[idx_t] + fw[idx_t]
                                                    + fi[idx_t] + fr[idx_t])))
            print("=" * 30)
            print(" Rho:   %.2e | %.2e" % (np.min(rho), np.max(rho)))
            print(" Vel:   %d | %d" % (np.min(1 / s), np.max(1 / s)))

            # Append response to existing data vectors
            t = np.append(t, self.RST.fop.response(s))
            rhoa = np.append(rhoa, self.ERT.fop.response(rho))

        # Return traveltimes and apparent resistivities for all time steps at once
        return pg.cat(t, rhoa)   