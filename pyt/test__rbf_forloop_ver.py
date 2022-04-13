import os, sys
import numpy   as np

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
    def rbf_func( xi, xj, coef=3.0 ):
        dist = np.sqrt( np.sum( ( xi - xj )**2, axis=-1 ) )
        rbf  = np.exp( -1.0 * coef*dist**2 )
        return( rbf )

    # ------------------------------------------------- #
    # --- [4] make G matrix                         --- #
    # ------------------------------------------------- #
    nTrain      = training_before.shape[0]
    Gmat        = np.zeros( (nTrain,nTrain) )
    for ik in range( nTrain ):
        for jk in range( nTrain ):
            Gmat[ik,jk] = rbf_func( training_before[ik,:], training_before[jk,:] )
    
    # ------------------------------------------------- #
    # --- [5] solve coefficient                     --- #
    # ------------------------------------------------- #
    if ( np.linalg.det( Gmat ) == 0.0 ):
        print( "regular matrix" )
        sys.exit()
    else:
        Ginv    = np.linalg.inv( Gmat )

    diffvect    = training_after[:,:] - training_before[:,:]
    alpha_x     = np.dot( Ginv, diffvect[:,x_] )
    alpha_y     = np.dot( Ginv, diffvect[:,y_] )
    alpha_z     = np.dot( Ginv, diffvect[:,z_] )
    
    # ------------------------------------------------- #
    # --- [6] interpolation                         --- #
    # ------------------------------------------------- #
    nPoints     = coord.shape[0]
    results     = np.zeros_like( coord )
    for ik in range( nPoints ):
        val_x, val_y, val_z = 0., 0., 0.
        for jk in range( nTrain ):
            val_x += alpha_x[jk] * rbf_func( coord[ik,:], training_before[jk,:] )
            val_y += alpha_y[jk] * rbf_func( coord[ik,:], training_before[jk,:] )
            val_z += alpha_z[jk] * rbf_func( coord[ik,:], training_before[jk,:] )
        results[ik,x_] = coord[ik,x_] + val_x
        results[ik,y_] = coord[ik,y_] + val_y
        results[ik,z_] = coord[ik,z_] + val_z

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
    fig.add__plot( xAxis=coord     [:,x_], yAxis=coord     [:,y_], color="Green", markersize=3.0 )
    fig.add__plot( xAxis=top_points[:,x_], yAxis=top_points[:,y_], color="slateblue", \
                   marker=None, linestyle="--" )
    fig.add__plot( xAxis=mod_points[:,x_], yAxis=mod_points[:,y_], color="Magenta"  , \
                   marker=None, linestyle="--" )
    fig.add__plot( xAxis=bot_points[:,x_], yAxis=bot_points[:,y_], color="slateblue", \
                   marker=None, linestyle="--" )
    fig.add__plot( xAxis=results   [:,x_], yAxis=results   [:,y_], color="Red" , markersize=3.0 )
    fig.set__axis()
    fig.save__figure()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    test()
