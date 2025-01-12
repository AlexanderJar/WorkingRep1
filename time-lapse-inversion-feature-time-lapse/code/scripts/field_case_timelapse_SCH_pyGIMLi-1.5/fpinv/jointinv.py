import pygimli as pg
from .lsqrinversion import LSQRInversion

class JointInv(LSQRInversion):
    def __init__(self, fop, data, error, startmodel, lam=20, beta=10000, 
                 delta=0, eps=0, atc=False,
                 maxIter=50, fwmin=0, fwmax=1, fimin=0, fimax=1, famin=0,
                 famax=1, frmin=0, frmax=1):
        LSQRInversion.__init__(self, data, fop, verbose=True, dosave=True)
        self._error = pg.Vector(error)

        # Check number of time steps
        ntimes = int(data.size() / (self.forwardOperator().RST.fop.data.size() 
                                    + self.forwardOperator().ERT.fop.data.size()))

        # Set data transformations
        self.logtrans = pg.trans.TransLog()
        self.trans = pg.trans.Trans()
        self.dcumtrans = pg.trans.TransCumulative()
        self.dcumtrans.add(self.trans,
                           self.forwardOperator().RST.fop.data.size() * ntimes)
        self.dcumtrans.add(self.logtrans,
                           self.forwardOperator().ERT.fop.data.size() * ntimes)
        self.setTransData(self.dcumtrans)

        # Set model transformation
        n = self.forwardOperator().cellCount
        self.mcumtrans = pg.trans.TransCumulative()
        self.transforms = []
        phase_limits = [[fwmin, fwmax], [fimin, fimax],
                        [famin, famax], [frmin, frmax]]
        for i, (lower, upper) in enumerate(phase_limits):
            if lower == 0:
                lower = 0.001
            self.transforms.append(pg.trans.TransLogLU(lower, upper))
            self.mcumtrans.add(self.transforms[i], n * ntimes)

        self.setTransModel(self.mcumtrans)

        # Set error
        self.setRelativeError(self._error)

        # Set some defaults
        # Set maximum number of iterations (default is 20)
        self.setMaxIter(maxIter)

        # Regularization strength (spatial smoothness)
        self.setLambda(lam)
        self.setDeltaPhiAbortPercent(0.25)

        fop = self.forwardOperator()
        fop.createConstraints(delta=delta, eps=eps, atc=atc)
        ones = pg.Vector(fop._I.rows(), 1.0)
        phiVec = pg.cat(ones, startmodel)
        self.setParameterConstraints(fop._G, phiVec, beta)
        self.setModel(startmodel)
