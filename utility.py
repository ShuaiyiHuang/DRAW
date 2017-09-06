import numpy as np
import matplotlib.pyplot as plt

def xrecons_grid(X,B,A):
	"""
	plots canvas for single time step
	X is x_recons, (batch_size x img_size)
	assumes features = BxA images
	batch is assumed to be a square number
	"""
	padsize=1
	padval=.5
	ph=B+2*padsize
	pw=A+2*padsize
	batch_size=X.shape[0]
	N=int(np.sqrt(batch_size))
	X=X.reshape((N,N,B,A))
	img=np.ones((N*ph,N*pw))*padval
	for i in range(N):
		for j in range(N):
			startr=i*ph+padsize
			endr=startr+B
			startc=j*pw+padsize
			endc=startc+A
			img[startr:endr,startc:endc]=X[i,j,:,:]
	return img

def save_image(x,T,B,A,count=0):
    for t in range(T):
        img = xrecons_grid(x[t],B,A)
        plt.matshow(img, cmap=plt.cm.gray)
        imgname = 'image/count_%d_%s_%d.png' % (count,'test', t)  # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
        plt.savefig(imgname)
        print(imgname)
