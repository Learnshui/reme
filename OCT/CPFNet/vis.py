import numpy as np
import visdom
import time

viz = visdom.Visdom(env="Test1") # 创建环境名为Test1
#单张图像显示与更新demo
image = viz.image(np.random.rand(3,256,256),opts={'title':'image1','caption':'How random.'})
# for i in range(10):
#     viz.image(np.random.randn( 3, 256, 256),win = image)
#     time.sleep(0.5)
#多图像显示与更新demo
images = viz.images(
        np.random.randn(20, 3, 64, 64),
        opts=dict(title='Random images', caption='How random.',nrow=5)
    )
for i in range(10):
    viz.images(np.random.randn(20, 3, 64, 64),win = images)
    time.sleep(0.5)