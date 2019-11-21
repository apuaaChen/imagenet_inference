import os


bs = [128, 64, 32, 16, 8, 4, 2, 1]

for b in bs:
    cmd = 'bash inference.sh mobilenet_v2 %d mobilenetv2_%d 16' % (b, b)
    os.system(cmd)
