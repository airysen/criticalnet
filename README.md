Critical Net
=====

# An open-source Python implementation of the critical net algorytm
based on *"Gu S., Zheng Y., Tomasi C. (2010) Critical Nets and Beta-Stable Features for Image Matching.
In: Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV
2010. ECCV 2010. Lecture Notes in Computer Science, vol 6313. Springer,
Berlin, Heidelberg"*
[http://link.springer.com/chapter/10.1007/978-3-642-15558-1_48](http://link.springer.com/chapter/10.1007/978-3-642-15558-1_48)

Installation
------------

Install the following requirements:

 * [scikit-image](scikit-image.org)
 * [NetworkX](https://networkx.github.io/)
 * [NumPy](http://numpy.org/)
 * [SciPy](http://scipy.org/)
 * [Matplotlib](http://matplotlib.org/)
 * [OpenCV-Python](http://opencv.org)

Example of usage
-----

```python
>>> import numpy as np
>>> import criticalnet as cnet
>>>
>>> im = cnet.data.sample('baboon')

>>> net0 = cnet.CriticalNet(image=im, ktimes=110, lap_mode='LOG')
>>> net0.compute(beta=10, extrema_dist=25, draw=True)
```

```python
>>> import numpy as np
>>> from scipy.misc import face 
>>> from PIL import Image
>>> import criticalnet as cnet
>>>
>>> racc = Image.fromarray(face(), mode='RGB')
>>> im = np.array(racc.convert(mode='F'))
>>>
>>> net0 = cnet.CriticalNet(image=im, ktimes=110)
>>> net0.calc_sscale()
>>> net0.calc_lap('DOG')
>>> net0.calc_beta_levels()
>>> net0.calc_cnet(beta=8, extrema_dist=30)
>>> net0.draw()
```

```python
>>> import networkx as nx
>>> G = net0.G
>>> nx.is_directed_acyclic_graph(G)
```
