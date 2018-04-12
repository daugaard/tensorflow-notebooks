import tensorflow as ts

sess = ts.Session()


hello = ts.constant("Hello World")

print(sess.run(hello))
