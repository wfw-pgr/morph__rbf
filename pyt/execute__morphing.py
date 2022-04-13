import numpy                       as np
import nkMeshRoutines.load__meshio as mio

# ========================================================= #
# ===  execute__morphing.py                             === #
# ========================================================= #

def execute__morphing():

    x_, y_, z_ = 0, 1, 2
    radius     = 1.050
    
    # ------------------------------------------------- #
    # --- [1] load msh File                         --- #
    # ------------------------------------------------- #
    mshFile        = "msh/model.msh"
    cells, points  = mio.load__meshio( mshFile=mshFile, returnType="elem-node", \
                                       elementType="tetra" )
    physnums       = mio.load__meshio( mshFile=mshFile, returnType="physnums", \
                                       elementType="tetra" )
    
    index_top      = np.where( points[:,z_] == 1.0 )
    index_mid      = np.where( points[:,z_] == 0.3 )
    index_bot      = np.where( points[:,z_] == 0.0 )
    boundaries_top = points[ index_top ]
    boundaries_mid = points[ index_mid ]
    boundaries_bot = points[ index_bot ]
    boundaries     = np.concatenate( [boundaries_top,boundaries_mid,boundaries_bot], axis=0 )

    radii          = np.sqrt( ( boundaries_mid[:,x_]**2 + boundaries_mid[:,y_]**2 ) )
    displacement_x = boundaries_mid[:,x_] * 0.0
    displacement_y = boundaries_mid[:,y_] * 0.0
    displacement_z = 0.15 * ( 1.0 - ( radii / radius )**2 )
    displace_mid   = np.concatenate( [displacement_x[:,None], displacement_y[:,None], \
                                      displacement_z[:,None] ], axis=-1 )
    displace_top   = np.zeros_like( boundaries_top )
    displace_bot   = np.zeros_like( boundaries_bot )
    displacement   = np.concatenate( [displace_top,displace_mid,displace_bot],axis=0 )
    
    import nkMeshRoutines.morph__rbf as mph
    updatept       = mph.morph__rbf( displacement=displacement, boundaries=boundaries, \
                                     nodes=points, coef=0.1 )
    import nkMeshRoutines.save__nastranFile as snf
    snf.save__nastranFile( points=updatept, cells=cells, outFile="out.bdf", matNums=physnums )

    return()



# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    execute__morphing()
