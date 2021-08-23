# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
from petsc4py import PETSc

class Petsc4pyMatContext:
    def __init__(self,op, mpi_comm=dl.mpi_comm_world()):
        self.op = op
        self.x = dl.PETScVector(mpi_comm)
        self.y = dl.PETScVector(mpi_comm)
        self.op.init_vector(self.x,1)
        self.op.init_vector(self.y,0)
         
    def mult(self, op, xx,yy):
        self.x.vec().aypx(0., xx)
        self.op.mult(self.x,self.y)
        yy.aypx(0., self.y.vec())
        
    def getSizes(self):
        return [dl.as_backend_type(self.y).vec().getSizes(),
                dl.as_backend_type(self.x).vec().getSizes()]
        
    def getComm(self):
        return dl.as_backend_type(self.x).vec().comm
         
class Petsc4pyPCContext:
    def __init__(self,prec, mpi_comm=dl.mpi_comm_world()):
        self.prec = prec
        self.x = dl.PETScVector(mpi_comm)
        self.b = dl.PETScVector(mpi_comm)
        self.prec.init_vector(self.x,1)
        self.prec.init_vector(self.b,0)
         
    def apply(self, prec, bb,xx):
        self.b.vec().aypx(0., bb)
        self.prec.solve(self.x,self.b)
        xx.aypx(0., self.x.vec())
        
    def getSizes(self):
        return [dl.as_backend_type(self.b).vec().getSizes(),
                dl.as_backend_type(self.x).vec().getSizes()]
        
def make_operator(A):
    context = Petsc4pyMatContext(A)
    Aop = PETSc.Mat().createPython(context.getSizes(), comm=context.getComm() )
    Aop.setPythonContext(context)
    Aop.setUp()
    
    return Aop

def setup_preconditioner(P, pc):
    contextpc = Petsc4pyPCContext(P)
    pc.setType(pc.Type.PYTHON)
    pc.setPythonContext(contextpc)
    pc.setUp()
