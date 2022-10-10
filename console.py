#coding: utf-8
from Tkinter import Tk, StringVar, Label, Entry, Button, W, E
from tkFileDialog import askopenfilename
from PIL import Image
from os import listdir, remove
from os.path import getsize
from math import ceil
from struct import pack, unpack
import tkMessageBox as mb
from hashlib import md5
from time import localtime
from numpy import mean
import matplotlib.pyplot as plt
from random import randint
from scipy import ndimage, misc

suffix = ''
filename = ''

class FileHead():
    def __init__(self, hash, idt, suf, total, num, size):
        self.hash = hash
        self.idt = idt
        self.suf = suf
        self.total = total
        self.num =num
        self.size = size

class FileSeg():
    def __init__(self, filehead, cont):
        self.filehead = filehead
        self.cont = cont

def disassemble(data):
    v = []
    # Pack file len in 4 bytes
    fSize = len(data)
    bytes = [ord(b) for b in pack("i", fSize)]
    bytes += [ord(b) for b in data]
    for b in bytes:
        for i in range(7, -1, -1):
            v.append((b >> i) & 0x1)
    '''
    blist = [ord(b) for b in f.read()]
    for b in blist:
        for i in xrange(8):
            bits.append((b >> i) & 1)
    '''
    return v

# Assemble an array of bits into a binary file
def assemble(v):
    bytes = ""
    length = len(v)
    for idx in range(0, len(v)/8):
        byte = 0
        for i in range(0, 8):
            if (idx*8+i < length):
                byte = (byte<<1) + v[idx*8+i]
        bytes = bytes + chr(byte)
    payload_size = unpack("I", bytes[:4])[0]
    return bytes[4: payload_size + 4]

# Set the i-th bit of v to x
def set_bit(n, i, x):
    mask = 1 << i
    n &= ~mask
    if x:
        n |= mask
    return n

def encodeDataToImages():#将文件隐写到图片中
    global filename, cooi, message, message1

    filecon = getsize(str(filename))*8# 单位为bit
    filenum = int(ceil(float(filecon)/cooi))
    images = listdir('./oldbmp')[:filenum]
    binaries = []

    with open(filename, 'rb') as frb:
        cont = disassemble(bytes(frb.read()))
    for i in range(filenum):
        if i != filenum-1:
            binaries.append(cont[i*cooi: (i+1)*cooi])
        else:
            binaries.append(cont[i*cooi:])

    b = listdir("./newbmp")#清空文件夹，准备存放新图片
    if len(b) != 0:
        for bmp in b:
            remove("./newbmp/" + bmp)

    def writehead(binary, i):
        payload = []#装填文件头和文件片段
        message1 = e_en.get()#获得密钥
        content = message1
        md5hash = md5(content)#加密
        h = md5hash.hexdigest()[:8]
        dmd5 = disassemble(h)#统一以比特流的格式
        payload += (dmd5)#文件头的第一部分（密钥哈希值）

        hour = localtime().tm_hour#获得时间，便于辨识
        md5hash = md5(str(hour))
        h = md5hash.hexdigest()[:4]
        hmd5 = disassemble(h)#统一以比特流的格式
        payload += (hmd5)#文件头的第二部分（时间标识码）

        suffix = filename[-4:]
        payload += (disassemble(pack("4s", bytes(suffix))))#文件后缀名的字符串，统一以比特流的格式
        payload +=(disassemble(pack("i", filenum)))#文件总分组数的字符串，统一以比特流的格式
        payload += (disassemble(pack("i", i)))#文件片段（子文件）序号，统一以比特流的格式

        n0 = len(binary)#最后一个字段，文件大小
        payload += (disassemble(pack("i", n0)))#单位bit，#统一以比特流的格式
        payload += binary#拼接秘密文件

        return payload#直接以比特流的格式

    def embed(imgFile, payload):
        # Process source image
        img = Image.open('./oldbmp/'+imgFile)
        conv = img.convert("RGB").getdata()
        v = payload
        # Add until multiple of 3
        while (len(v) % 3):
            v.append(0)

        oldencodedPixels = []  # 将 image 中的像素保存下来
        for i, (r, g, b) in enumerate(list(conv)):
            oldencodedPixels.append((r, g, b))
        encodedImage = Image.new(img.mode, img.size)  # 创建新图片以存放编码后的像素
        encodedImage.putdata(oldencodedPixels)
        idx = 0
        encodedPixels = []  # 将 binary 中的二进制字符串信息编码进像素里
        for i, (r, g, b) in enumerate(list(conv)):
            if idx < len(v):
                r0 = set_bit(r, 0, v[idx])
                g0 = set_bit(g, 0, v[idx + 1])
                b0 = set_bit(b, 0, v[idx + 2])
                encodedPixels.append((r0, g0, b0))
                idx = idx + 3
        encodedImage.putdata(encodedPixels)
        encodedImage.save('./newbmp/'+imgFile)

    for i in range(filenum):#filenum即子文件（或曰文件片段）的总数
        embed(images[i], writehead(binaries[i], i))
    mb.showinfo('提示', '隐写成功，图片已保存到newbmp路径下')


def decodeDataInImages():#解码隐藏数据
    message2 = e_de.get()
    images = listdir('./newbmp')
    ilen = len(images)
    data = []
    filesegs = []

    def extracthead(in_file):
        content = message2#获得用户输入的解密密钥
        md5hash = md5(content)
        h = md5hash.hexdigest()[:8]
        dmd5 = disassemble(h)#进行hash处理

        img = Image.open('./newbmp/'+in_file)
        conv = img.convert("RGB").getdata()#获得图片的像素点
        # Extract LSBs
        v = []#装载文件头
        v0 = []#装载文件内容
        headlen = 416
        idx = 0

        for i, (r, g, b) in enumerate(list(conv)):
            if idx < headlen and i<headlen:
                v.append(r & 1)
                v.append(g & 1)
                v.append(b & 1)
            idx += 3
        v = v[:headlen]
        filehead = FileHead(assemble(v[:96]), assemble(v[96:160]),
                            assemble(v[160:224]),
                            unpack('i',assemble(v[224:288]))[0],
                            unpack('i',assemble(v[288:352]))[0],
                            unpack('i', assemble(v[352:416]))[0])
        dhead = filehead.hash

        if dhead != assemble(dmd5):
            return

        limit = filehead.size#文件头中记录的的子文件大小
        idx = 0
        for i1, (r, g, b) in enumerate(list(conv)):
            if idx < limit:#超出子文件大小则无需再读取
                v0.append(r & 1)
                v0.append(g & 1)
                v0.append(b & 1)
                idx = idx + 3
            else:
                break
        # 第一个参数为文件头，第二个为子文件内容
        return FileSeg(filehead, v0[416:])

    for i in range(ilen):
        tmp = extracthead(images[i])
        if tmp != None:
            filesegs.append(tmp)
        else:
            break

    if filesegs != []:
        for i in range(ilen):#根据子文件序号排序
            for j in range(ilen):
                if filesegs[j].filehead.num == i:
                    data += filesegs[j].cont#将各子文件的内容按序拼接
    suffix = filesegs[0].filehead.suf
    if suffix[0] != '.':
        suffix = '.' + suffix
    if filesegs != [] and data != []:
        with open("restoredcipher"+suffix, "wb") as rcf:
            rcf.write(assemble(data))
        mb.showinfo('提示', '提取成功，文件已还原到本路径的restoredcipher中')
    else:
        mb.showinfo('提示', '密钥错误')

def analyseDataInImages():
    bmps = listdir('analysis')
    for i0 in range(len(bmps)):
        BS = 2000  # Block size
        img = Image.open('./analysis/' + bmps[i0])
        (width, height) = img.size
        conv = img.convert("RGB").getdata()

        # Extract LSBs
        vr = []  # Red LSBs
        vg = []  # Green LSBs
        vb = []  # LSBs
        for h in range(height):
            for w in range(width):
                (r, g, b) = conv.getpixel((w, h))
                vr.append(r & 1)
                vg.append(g & 1)
                vb.append(b & 1)

        # 包含平均值
        avgR = []
        avgG = []
        avgB = []
        for i in range(0, len(vr), BS):# 将图片分块读取
            avgR.append(mean(vr[i:i + BS]))# 计算每个channel的LSB位的平均值
            avgG.append(mean(vg[i:i + BS]))
            avgB.append(mean(vb[i:i + BS]))

        # Nice plot
        numBlocks = len(avgR)
        blocks = [i for i in range(0, numBlocks)]
        plt.axis([0, len(avgR), 0, 1])
        # 若是被隐写，则平均值应该接近0.5
        plt.ylabel('Average LSB per block')
        plt.xlabel('Block number')
        plt.figure(num=i0+1)
        plt.plot(blocks, avgR, 'r.')
        plt.plot(blocks, avgG, 'g')
        plt.plot(blocks, avgB, 'b')

    plt.show()

def matchingencodeDataToImages():
    global filename, cooi, message, message1

    filecon = getsize(str(filename)) * 8  # 单位为bit
    filenum = int(ceil(float(filecon) / cooi))
    images = listdir('./matching/oldbmp')[:filenum]
    binaries = []

    with open(filename, 'rb') as frb:
        cont = disassemble(bytes(frb.read()))
    for i in range(filenum):
        if i != filenum - 1:
            binaries.append(cont[i * cooi: (i + 1) * cooi])
        else:
            binaries.append(cont[i * cooi:])

    b = listdir("./matching/newbmp")  # 清空文件夹，准备存放新图片
    if len(b) != 0:
        for bmp in b:
            remove("./matching/newbmp/" + bmp)

    def matchingwritehead(binary, i):
        payload = []  # 装填文件头和文件片段
        message1 = e_en.get()  # 获得密钥
        content = message1
        md5hash = md5(content)  # 加密
        h = md5hash.hexdigest()[:8]
        dmd5 = disassemble(h)  # 统一以比特流的格式
        payload += (dmd5)  # 文件头的第一部分（密钥哈希值）

        hour = localtime().tm_hour  # 获得时间，便于辨识
        md5hash = md5(str(hour))
        h = md5hash.hexdigest()[:4]
        hmd5 = disassemble(h)  # 统一以比特流的格式
        payload += (hmd5)  # 文件头的第二部分（时间标识码）

        suffix = filename[-4:]
        payload += (disassemble(pack("4s", bytes(suffix))))  # 文件后缀名的字符串，统一以比特流的格式
        payload += (disassemble(pack("i", filenum)))  # 文件总分组数的字符串，统一以比特流的格式
        payload += (disassemble(pack("i", i)))  # 文件片段（子文件）序号，统一以比特流的格式

        n0 = len(binary)  # 最后一个字段，文件大小
        payload += (disassemble(pack("i", n0)))  # 单位bit，#统一以比特流的格式
        payload += binary

        return payload  # 直接以比特流的格式

    def matchingembed(imgFile, payload):
        # Process source image
        I = misc.imread('./matching/oldbmp/' + imgFile)
        v = payload
        # Add until multiple of 3
        sign = [1, -1]
        idx = 0
        for i in xrange(I.shape[0]):
            for j in xrange(I.shape[1]):
                for k in xrange(3):
                    if idx < len(v):
                        if I[i][j][k] % 2 != v[idx]:
                            s = sign[randint(0, 1)]
                            if I[i][j][k] == 0: s = 1
                            if I[i][j][k] == 255: s = -1
                            I[i][j][k] += s
                        idx += 1
        misc.imsave('./matching/newbmp/' + imgFile, I)
    for i in range(filenum):  # filenum即子文件（或曰文件片段）的总数
        matchingembed(images[i], matchingwritehead(binaries[i], i))
    mb.showinfo('提示', '隐写成功，图片已保存到matching/newbmp路径下')

def matchingdecodeDataInImages():#解码隐藏数据
    message2 = e_de.get()
    images = listdir('./matching/newbmp')
    ilen = len(images)
    data = []
    filesegs = []

    def extracthead(in_file):
        content = message2#获得用户输入的解密密钥
        md5hash = md5(content)
        h = md5hash.hexdigest()[:8]
        dmd5 = disassemble(h)#进行hash处理

        img = Image.open('./matching/newbmp/'+in_file)
        conv = img.convert("RGB").getdata()#获得图片的像素点
        # Extract LSBs
        v = []#装载文件头
        v0 = []#装载文件内容
        headlen = 416
        idx = 0

        for i, (r, g, b) in enumerate(list(conv)):
            if idx < headlen and i<headlen:
                v.append(r & 1)
                v.append(g & 1)
                v.append(b & 1)
            idx += 3
        v = v[:headlen]
        filehead = FileHead(assemble(v[:96]), assemble(v[96:160]),
                            assemble(v[160:224]),
                            unpack('i',assemble(v[224:288]))[0],
                            unpack('i',assemble(v[288:352]))[0],
                            unpack('i', assemble(v[352:416]))[0])
        dhead = filehead.hash

        if dhead != assemble(dmd5):
            return

        limit = filehead.size#文件头中记录的的子文件大小
        idx = 0
        for i1, (r, g, b) in enumerate(list(conv)):
            if idx < limit:#超出子文件大小则无需再读取
                v0.append(r & 1)
                v0.append(g & 1)
                v0.append(b & 1)
                idx = idx + 3
            else:
                break
        # 第一个参数为文件头，第二个为子文件内容
        return FileSeg(filehead, v0[416:])

    for i in range(ilen):
        tmp = extracthead(images[i])
        if tmp != None:
            filesegs.append(tmp)
        else:
            break

    if filesegs != []:
        for i in range(ilen):#根据子文件序号排序
            for j in range(ilen):
                if filesegs[j].filehead.num == i:
                    data += filesegs[j].cont#将各子文件的内容按序拼接
    suffix = filesegs[0].filehead.suf
    if suffix[0] != '.':
        suffix = '.' + suffix
    if filesegs != [] and data != []:
        with open("./matching/restoredcipher"+suffix, "wb") as rcf:
            rcf.write(assemble(data))
        mb.showinfo('提示', '提取成功，文件已还原到/matching的restoredcipher中')
    else:
        mb.showinfo('提示', '密钥错误')


def selectPath():
    global filename
    path_ = askopenfilename()
    path.set(path_)
    filename = path_
    return path_

root = Tk()#窗口框架
root.resizable(width = 100, height = 300)
root.title("大文件隐写处理")
path = StringVar()
strpath=" "

#第0行，选择文件
l_fi =Label(root,text='目标文件:')
l_fi.grid(row=0,sticky=W)
e_fi =Entry(root, textvariable=path, width=30)
e_fi.grid(row=0,column=1,sticky=E)
Button(root, text="文件选择", command=selectPath).grid(row = 0, column = 2)

# 第1行，输入加密密钥
l_en = Label(root, text='请输入加密密钥：')
l_en.grid(row=1, sticky=W)
e_en = Entry(root)
e_en.grid(row=1, column=1, sticky=E)

# 第2行，输入解密密钥
l_de = Label(root, text='请输入解密密钥')
l_de.grid(row=2, sticky=W)
e_de = Entry(root)
#e_de['show'] = '*'
e_de.grid(row=2, column=1, sticky=E)

imagesarr = listdir('./oldbmp')
img = Image.open('./oldbmp/'+imagesarr[0])
cooi = img.width * img.height * len(img.split())-416# content of one image，一幅图片容纳的信息量,减去约定的文件头数据

bt2 = Button(root, text="隐藏信息", width=15, height=1, command=encodeDataToImages).grid(row=3, column=1)
bt3 = Button(root, text="提取信息", width=15, height=1, command=decodeDataInImages).grid(row=4, column=1)
bt4 = Button(root, text="隐写分析", width=15, height=1, command=analyseDataInImages).grid(row=5, column=1)
bt5 = Button(root, text="隐蔽隐藏", width=15, height=1, command=matchingencodeDataToImages).grid(row=6, column=1)
bt6 = Button(root, text="隐蔽提取", width=15, height=1, command=matchingdecodeDataInImages).grid(row=7, column=1)

root.mainloop()