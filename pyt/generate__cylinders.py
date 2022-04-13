import numpy as np
import os, sys
import gmsh

# ========================================================= #
# ===  geometry                                    === #
# ========================================================= #

def make__geometry():

    radius  = 1.050
    dz1     = 0.3
    dz2     = 0.7
    
    zFrom1  = 0.0
    zFrom2  = zFrom1 + dz1
    tag1    = gmsh.model.occ.addCylinder( 0,0,zFrom1, 0,0,dz1, radius )
    tag2    = gmsh.model.occ.addCylinder( 0,0,zFrom2, 0,0,dz2, radius )
    dimtags = { "cyl01":[(3,tag1)], "cyl02":[(3,tag2)] }
    print( tag1, tag2 ) 
    
    return( dimtags )

    


# ========================================================= #
# ===   実行部                                          === #
# ========================================================= #

if ( __name__=="__main__" ):
    

    # ------------------------------------------------- #
    # --- [1] initialization of the gmsh            --- #
    # ------------------------------------------------- #
    gmsh.initialize()
    gmsh.option.setNumber( "General.Terminal", 1 )
    gmsh.option.setNumber( "Mesh.Algorithm"  , 5 )
    gmsh.option.setNumber( "Mesh.Algorithm3D", 4 )
    gmsh.option.setNumber( "Mesh.SubdivisionAlgorithm", 0 )
    gmsh.model.add( "model" )
    
    
    # ------------------------------------------------- #
    # --- [2] Modeling                              --- #
    # ------------------------------------------------- #

    dimtags = make__geometry()
    
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()


    # ------------------------------------------------- #
    # --- [3] Mesh settings                         --- #
    # ------------------------------------------------- #
    meshFile = "dat/mesh.conf"
    physFile = "dat/phys.conf"
    import nkGmshRoutines.assign__meshsize as ams
    meshes = ams.assign__meshsize( dimtags=dimtags, meshFile=meshFile, physFile=physFile )

    gmsh.option.setNumber( "Mesh.CharacteristicLengthMin", 0.1 )
    gmsh.option.setNumber( "Mesh.CharacteristicLengthMax", 0.1 )
    

    # ------------------------------------------------- #
    # --- [4] post process                          --- #
    # ------------------------------------------------- #
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write( "msh/model.msh" )
    gmsh.finalize()
