import os, sys
import numpy as np

# ========================================================= #
# ===  morphing using RBF                               === #
# ========================================================= #

def morph__rbf( boundaries=None, displacement=None, nodes=None, rbfType="gaussian", coef=1.0 ):

    x_, y_, z_ = 0, 1, 2
    
    # ------------------------------------------------- #
    # --- [1] Arguments                             --- #
    # ------------------------------------------------- #
    if ( boundaries   is None ): sys.exit( "[morph__rbf.py] boundaries   == ???" )
    if ( nodes        is None ): sys.exit( "[morph__rbf.py] nodes        == ???" )
    if ( displacement is None ): sys.exit( "[morph__rbf.py] displacement == ???" )
    
    # ------------------------------------------------- #
    # --- [2] define rbf : gaussian                 --- #
    # ------------------------------------------------- #
    def rbf__gaussian( xi, xj, coef=0.5 ):
        dist  = np.sqrt( np.sum( ( xi - xj )**2, axis=-1 ) )
        value = np.exp( - ( dist / coef )**2 )
        return( value )

    # ------------------------------------------------- #
    # --- [3] function to be used                   --- #
    # ------------------------------------------------- #
    if   ( rbfType.lower() == "gaussian" ):
        rbf_func = rbf__gaussian
    else:
        print( "[morph__rbf] unknown rbf Kernel Type :: {0} ".format( rbfType ) )
        sys.exit()
        
    # ------------------------------------------------- #
    # --- [4] make G matrix                         --- #
    # ------------------------------------------------- #
    nTrain      = boundaries.shape[0]
    xvec1,xvec2 = np.meshgrid( boundaries[:,x_], boundaries[:,x_], indexing="ij" )
    yvec1,yvec2 = np.meshgrid( boundaries[:,y_], boundaries[:,y_], indexing="ij" )
    zvec1,zvec2 = np.meshgrid( boundaries[:,z_], boundaries[:,z_], indexing="ij" )
    tvec1       = np.concatenate( [ xvec1[:,:,None],yvec1[:,:,None],zvec1[:,:,None] ], axis=2 )
    tvec2       = np.concatenate( [ xvec2[:,:,None],yvec2[:,:,None],zvec2[:,:,None] ], axis=2 )
    Gmat        = rbf_func( tvec1, tvec2, coef=coef )
    Ginv        = np.linalg.inv( Gmat )
    
    # ------------------------------------------------- #
    # --- [5] solve coefficient                     --- #
    # ------------------------------------------------- #
    alphas      = np.dot( Ginv, displacement )
    
    # ------------------------------------------------- #
    # --- [6] interpolation                         --- #
    # ------------------------------------------------- #
    xvec1,xvec2 = np.meshgrid( nodes[:,x_], boundaries[:,x_], indexing="ij" )
    yvec1,yvec2 = np.meshgrid( nodes[:,y_], boundaries[:,y_], indexing="ij" )
    zvec1,zvec2 = np.meshgrid( nodes[:,z_], boundaries[:,z_], indexing="ij" )
    tvec1       = np.concatenate( [ xvec1[:,:,None],yvec1[:,:,None],zvec1[:,:,None] ], axis=2 )
    tvec2       = np.concatenate( [ xvec2[:,:,None],yvec2[:,:,None],zvec2[:,:,None] ], axis=2 )
    Rmat        = rbf_func( tvec1, tvec2, coef=coef )
    results     = nodes + np.dot( Rmat, alphas )

    return( results )

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    x_, y_, z_ = 0, 1, 2
    
    # ------------------------------------------------- #
    # --- [1] coordinate                            --- #
    # ------------------------------------------------- #
    import nkUtilities.equiSpaceGrid as esg
    x1MinMaxNum = [ 0.0, 1.0, 11 ]
    x2MinMaxNum = [ 0.0, 1.0, 11 ]
    x3MinMaxNum = [ 0.0, 0.0,  1 ]
    nodes       = esg.equiSpaceGrid( x1MinMaxNum=x1MinMaxNum, x2MinMaxNum=x2MinMaxNum, \
                                     x3MinMaxNum=x3MinMaxNum, returnType = "point" )
    
    # ------------------------------------------------- #
    # --- [2] training nodes                        --- #
    # ------------------------------------------------- #
    bot_points        = np.zeros( (11,3) )
    bot_points[:,x_]  = np.linspace( 0.0, 1.0, 11 )
    bot_points[:,y_]  = 0.0
    top_points        = np.zeros( (11,3) )
    top_points[:,x_]  = np.linspace( 0.0, 1.0, 11 )
    top_points[:,y_]  = 1.0
    mod_points        = np.zeros( (11,3) )
    mod_points[:,x_]  = np.linspace( 0.0, 1.0, 11 )
    mod_points[:,y_]  = 1.0 - 0.2*np.sin( mod_points[:,x_] * np.pi )

    boundaries_before = np.concatenate( [top_points,bot_points], axis=0 )
    boundaries_after  = np.concatenate( [mod_points,bot_points], axis=0 )

    displacement      = boundaries_after - boundaries_before
    
    results           = morph__rbf( displacement=displacement, boundaries=boundaries_before, \
                                    nodes=nodes, rbfType="gaussian", coef=0.5 )

    # ------------------------------------------------- #
    # --- [7] plot points                           --- #
    # ------------------------------------------------- #
    import nkUtilities.plot1D         as pl1
    import nkUtilities.load__config   as lcf
    import nkUtilities.configSettings as cfs
    x_,y_                    = 0, 1
    pngFile                  = "test/out.png"
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["plt_xAutoRange"] = False
    config["plt_yAutoRange"] = False
    config["plt_xRange"]     = [ -0.2, +1.2 ]
    config["plt_yRange"]     = [ -0.2, +1.2 ]
    config["plt_linestyle"]  = "none"
    config["plt_marker"]     = "o"
    
    fig     = pl1.plot1D( config=config, pngFile=pngFile )
    fig.add__plot( xAxis=nodes     [:,x_], yAxis=nodes     [:,y_], color="Green" )
    fig.add__plot( xAxis=top_points[:,x_], yAxis=top_points[:,y_], color="slateblue", \
                   marker=None, linestyle="--" )
    fig.add__plot( xAxis=mod_points[:,x_], yAxis=mod_points[:,y_], color="Magenta"  , \
                   marker=None, linestyle="--" )
    fig.add__plot( xAxis=bot_points[:,x_], yAxis=bot_points[:,y_], color="slateblue", \
                   marker=None, linestyle="--" )
    fig.add__plot( xAxis=results   [:,x_], yAxis=results   [:,y_], color="Red" )
    fig.set__axis()
    fig.save__figure()

