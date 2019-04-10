def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        '''
        构建YOLO网络
        
        args:
            images：输入图片占位符 [None,image_size,image_size,3]  这里是[None,448,448,3]
            num_outputs：标量，网络输出节点数 1470
            alpha：泄露修正线性激活函数 系数0.1
            keep_prob：弃权 保留率
            is_training：训练？
            scope：命名空间名
            
        return：
            返回网络最后一层，激活函数处理之前的值  形状[None,1470]
        '''
        #定义变量命名空间
        with tf.variable_scope(scope):
            #定义共享参数  使用l2正则化
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                logging.info('image shape{0}'.format(images.shape))
                #pad_1 填充 454x454x3
                net = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
                logging.info('Layer pad_1  {0}'.format(net.shape))
                #卷积层conv_2 s=2    (n-f+1)/s向上取整    224x224x64
                net = slim.conv2d(
                    net, 64, 7, 2, padding='VALID', scope='conv_2')         
                logging.info('Layer conv_2 {0}'.format(net.shape))
                #池化层pool_3 112x112x64
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                logging.info('Layer pool_3 {0}'.format(net.shape))
                #卷积层conv_4、3x3x192 s=1  n/s向上取整   112x112x192
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                logging.info('Layer conv_4 {0}'.format(net.shape))
                #池化层pool_5 56x56x192
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                logging.info('Layer pool_5 {0}'.format(net.shape))
                #卷积层conv_6、1x1x128 s=1  n/s向上取整  56x56x128
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                logging.info('Layer conv_6 {0}'.format(net.shape))
                #卷积层conv_7、3x3x256 s=1  n/s向上取整 56x56x256
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                logging.info('Layer conv_7 {0}'.format(net.shape))
                #卷积层conv_8、1x1x256 s=1  n/s向上取整 56x56x256
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                logging.info('Layer conv_8 {0}'.format(net.shape))
                #卷积层conv_9、3x3x512 s=1  n/s向上取整 56x56x512
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                logging.info('Layer conv_9 {0}'.format(net.shape))
                #池化层pool_10 28x28x512
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                logging.info('Layer pool_10 {0}'.format(net.shape))
                #卷积层conv_11、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                logging.info('Layer conv_11 {0}'.format(net.shape))
                #卷积层conv_12、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                logging.info('Layer conv_12 {0}'.format(net.shape))
                #卷积层conv_13、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                logging.info('Layer conv_13 {0}'.format(net.shape))
                #卷积层conv_14、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                logging.info('Layer conv_14 {0}'.format(net.shape))
                #卷积层conv_15、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                logging.info('Layer conv_15 {0}'.format(net.shape))                
                #卷积层conv_16、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                logging.info('Layer conv_16 {0}'.format(net.shape))
                #卷积层conv_17、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                logging.info('Layer conv_17 {0}'.format(net.shape))                
                #卷积层conv_18、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                logging.info('Layer conv_18 {0}'.format(net.shape))
                #卷积层conv_19、1x1x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                logging.info('Layer conv_19 {0}'.format(net.shape))
                #卷积层conv_20、3x3x1024 s=1  n/s向上取整 28x28x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                logging.info('Layer conv_20 {0}'.format(net.shape))
                #池化层pool_21 14x14x1024
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                logging.info('Layer pool_21 {0}'.format(net.shape))
                #卷积层conv_22、1x1x512 s=1  n/s向上取整 14x14x512
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                logging.info('Layer conv_22 {0}'.format(net.shape))
                #卷积层conv_23、3x3x1024 s=1  n/s向上取整 14x14x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                logging.info('Layer conv_23 {0}'.format(net.shape))
                #卷积层conv_24、1x1x512 s=1  n/s向上取整 14x14x512
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                logging.info('Layer conv_24 {0}'.format(net.shape))
                #卷积层conv_25、3x3x1024 s=1  n/s向上取整 14x14x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                logging.info('Layer conv_25 {0}'.format(net.shape))
                #卷积层conv_26、3x3x1024 s=1  n/s向上取整 14x14x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                logging.info('Layer conv_26 {0}'.format(net.shape))
                #pad_27 填充 16x16x2014
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                logging.info('Layer pad_27 {0}'.format(net.shape))
                #卷积层conv_28、3x3x1024 s=2  (n-f+1)/s向上取整 7x7x1024
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')
                logging.info('Layer conv_28 {0}'.format(net.shape))
                #卷积层conv_29、3x3x1024 s=1  n/s向上取整 7x7x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                logging.info('Layer conv_29 {0}'.format(net.shape))
                #卷积层conv_30、3x3x1024 s=1  n/s向上取整 7x7x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                logging.info('Layer conv_30 {0}'.format(net.shape))
                #trans_31 转置[None,1024,7,7]
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                logging.info('Layer trans_31 {0}'.format(net.shape))
                #flat_32 展开 50176
                net = slim.flatten(net, scope='flat_32')
                logging.info('Layer flat_32 {0}'.format(net.shape))
                #全连接层fc_33  512
                net = slim.fully_connected(net, 512, scope='fc_33')
                logging.info('Layer fc_33 {0}'.format(net.shape))
                #全连接层fc_34  4096
                net = slim.fully_connected(net, 4096, scope='fc_34')
                logging.info('Layer fc_34 {0}'.format(net.shape))
                #弃权层dropout_35 4096
                net = slim.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                logging.info('Layer dropout_35 {0}'.format(net.shape))
                #全连接层fc_36 1470
                net = slim.fully_connected(
                    net, num_outputs, activation_fn=None, scope='fc_36')
                logging.info('Layer fc_36 {0}'.format(net.shape))
        return net
