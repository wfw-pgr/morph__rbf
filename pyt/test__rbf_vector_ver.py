import os, sys
import numpy as np

# ========================================================= #
# ===  test ( rbf interpolation )                       === #
# ========================================================= #

def test():

    x_, y_, z_ = 0, 1, 2
    
    # ------------------------------------------------- #
    # --- [1] coordinate                            --- #
    # ------------------------------------------------- #
    import nkUtilities.equiSpaceGrid as esg
    x1MinMaxNum = [ 0.0, 1.0, 11 ]
    x2MinMaxNum = [ 0.0, 1.0, 11 ]
    x3MinMaxNum = [ 0.0, 0.0,  1 ]
    coord       = esg.equiSpaceGrid( x1MinMaxNum=x1MinMaxNum, x2MinMaxNum=x2MinMaxNum, \
                                     x3MinMaxNum=x3MinMaxNum, returnType = "point" )
    
    # ------------------------------------------------- #
    # --- [2] training nodes                        --- #
    # ------------------------------------------------- #
    bot_points       = np.zeros( (11,3) )
    bot_points[:,x_] = np.linspace( 0.0, 1.0, 11 )
    bot_points[:,y_] = 0.0
    top_points       = np.zeros( (11,3) )
    top_points[:,x_] = np.linspace( 0.0, 1.0, 11 )
    top_points[:,y_] = 1.0
    mod_points       = np.zeros( (11,3) )
    mod_points[:,x_] = np.linspace( 0.0, 1.0, 11 )
    mod_points[:,y_] = 1.0 - 0.2*np.sin( mod_points[:,x_] * np.pi )

    training_before  = np.concatenate( [top_points,bot_points], axis=0 )
    training_after   = np.concatenate( [mod_points,bot_points], axis=0 )

    # ------------------------------------------------- #
    # --- [3] define rbf                            --- #
    # ------------------------------------------------- #
    def rbf_func( xi, xj, coef=0.5 ):
        dist = np.sqrt( np.sum( ( xi - xj )**2, axis=-1 ) )
        rbf  = np.exp( - ( dist / coef )**2 )
        return( rbf )

    # ------------------------------------------------- #
    # --- [4] make G matrix                         --- #
    # ------------------------------------------------- #
    nTrain      = training_before.shape[0]
    xvec1,xvec2 = np.meshgrid( training_before[:,x_], training_before[:,x_], indexing="ij" )
    yvec1,yvec2 = np.meshgrid( training_before[:,y_], training_before[:,y_], indexing="ij" )
    zvec1,zvec2 = np.meshgrid( training_before[:,z_], training_before[:,z_], indexing="ij" )
    tvec1       = np.concatenate( [ xvec1[:,:,None],yvec1[:,:,None],zvec1[:,:,None] ], axis=2 )
    tvec2       = np.concatenate( [ xvec2[:,:,None],yvec2[:,:,None],zvec2[:,:,None] ], axis=2 )
    Gmat        = rbf_func( tvec1, tvec2 )
    Ginv        = np.linalg.inv( Gmat )
    
    # ------------------------------------------------- #
    # --- [5] solve coefficient                     --- #
    # ------------------------------------------------- #
    displaces   = training_after - training_before
    alphas      = np.dot( Ginv, displaces )
    print( alphas.shape, displaces.shape, Ginv.shape )
    
    # ------------------------------------------------- #
    # --- [6] interpolation                         --- #
    # ------------------------------------------------- #
    nPoints     = coord.shape[0]
    xvec1,xvec2 = np.meshgrid( coord[:,x_], training_before[:,x_], indexing="ij" )
    yvec1,yvec2 = np.meshgrid( coord[:,y_], training_before[:,y_], indexing="ij" )
    zvec1,zvec2 = np.meshgrid( coord[:,z_], training_before[:,z_], indexing="ij" )
    tvec1       = np.concatenate( [ xvec1[:,:,None],yvec1[:,:,None],zvec1[:,:,None] ], axis=2 )
    tvec2       = np.concatenate( [ xvec2[:,:,None],yvec2[:,:,None],zvec2[:,:,None] ], axis=2 )
    Rmat        = rbf_func( tvec1, tvec2 )
    results     = np.zeros_like( coord )
    results     = coord + np.dot( Rmat, alphas )
    # results[:,x_] = coord[:,x_] + np.dot( Rmat, alpha_x )
    # results[:,y_] = coord[:,y_] + np.dot( Rmat, alpha_y )
    # results[:,z_] = coord[:,z_] + np.dot( Rmat, alpha_z )

    # ------------------------------------------------- #
    # --- [7] plot points                           --- #
    # ------------------------------------------------- #
    import nkUtilities.plot1D         as pl1
    import nkUtilities.load__config   as lcf
    import nkUtilities.configSettings as cfs
    x_,y_                    = 0, 1
    pngFile                  = "png/out.png"
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["plt_xAutoRange"] = False
    config["plt_yAutoRange"] = False
    config["plt_xRange"]     = [ -0.2, +1.2 ]
    config["plt_yRange"]     = [ -0.2, +1.2 ]
    config["plt_linestyle"]  = "none"
    config["plt_marker"]     = "o"
    
    fig     = pl1.plot1D( config=config, pngFile=pngFile )
    fig.add__plot( xAxis=coord     [:,x_], yAxis=coord     [:,y_], color="Green" )
    fig.add__plot( xAxis=top_points[:,x_], yAxis=top_points[:,y_], color="slateblue", \
                   marker=None, linestyle="--" )
    fig.add__plot( xAxis=mod_points[:,x_], yAxis=mod_points[:,y_], color="Magenta"  , \
                   marker=None, linestyle="--" )
    fig.add__plot( xAxis=bot_points[:,x_], yAxis=bot_points[:,y_], color="slateblue", \
                   marker=None, linestyle="--" )
    fig.add__plot( xAxis=results   [:,x_], yAxis=results   [:,y_], color="Red" )
    fig.set__axis()
    fig.save__figure()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    test()
