```相对的改进
#---------------------------(predict.ipynb)------------------------#
    if mode == "predict":
        image = Image.open('4.jpg')#先自动对目录下的4.jpg文件实施基线预测
        #此外还提供的基准图片有：45是基线目标，67是大目标，89是小目标，cd是难目标
        r_image = yolo.detect_image(image, crop = crop, count=count)
        r_image.show()
        while True:
            img = input('Input image filename:')
            try:#这样自动叠加后缀就只需要输入文件名
                image = Image.open(img+'.jpg')
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()
#---------------------------(yolo.py)------------------------#
        "model_path"        : 'model_data/b基础633.pth',#原yolov8_s换为自训的基线权
        "classes_path"      : 'model_data/voc_classes.txt',#只含0到6七类，分别分行
        "phi"               : 'n',#版本从s换为更易训、内存更小的n 
        "cuda"              : False,#cuda换为否方便推理时切无卡模式用cpu更省钱
#---------------------------(utils_fit.py)------------------------#
    if local_rank == 0:#去掉开训和完训，以及验证全程的显示
        # print('Start Train')
    if local_rank == 0:
        pbar.close()
        # print('Finish Train')
        # print('Start Validation')
        # pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    if local_rank == 0:
        pbar.close()
        # print('Finish Validation')
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):#关掉最优权的保存提示，将定期权重名改为p030三个数的形式，忽略具体损失，最后精简best_epoch_weights为b，last_epoch_weights为l
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            # print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "b.pth"))
        #     torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        # torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "l.pth"))
#---------------------------(callbacks.py)------------------------#
            # print("Calculate Map.")
            # print("Get map done.") #关掉算map始末的提示
#---------------------------(train.ipynb)------------------------#
if __name__ == "__main__": #精简参数行，去除多余注释
    Cuda            = True #服务器训练只能用gpu，无卡模式cpu训不了
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = True #设true更快些
    classes_path    = 'model_data/voc_classes.txt'
    model_path      = 'b基础633.pth' #原为'model_data/yolov8_s.pth'改成咱们自训的
    input_shape     = [640, 640]
    phi             = 'n' # 原's'改更小更高效
    pretrained      = False #有权重就不用预训练
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    label_smoothing     = 0
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2 #原32改小
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 4 #原16改小
    Freeze_Train        = False #预冻结前50的骨网权重，在前置网需要同时训练故设False
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = "cos"
    save_period         = 30 #每隔30轮保存下权重，整个只需10个文件，减少原10的冗余
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 10
    num_workers         = 4
```
检测头部分
```
# ------------------------(yolo.py)-----------------------------#
class YoloBody(nn.Module):
        # self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, 64, 3), Conv(64, 64, 3), nn.Conv2d(64, 8, 1)) for x in ch)
        for i in range(self.nl):
         x[i]=torch.cat((self.cv2[i](x[i]),self.cv3[i](x[i]),self.cv4[i](x[i])),1)
        # if self.shape != shape:
        box,cls, ang        = torch.cat([xi.view(shape[0], self.no+8, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes,8), 1)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device), ang #需去距离头

# ------------------------(utils_bbox)---------------------------------#
class DecodeBox():
    def decode_box(self, inputs): #就是补充",ang"
        dbox, cls, origin_cls, anchors, strides, ang = inputs
        y       = torch.cat((dbox, cls.sigmoid(), ang), 1).permute(0, 2, 1)

    def 相交相似(self,当框,另框):#适当放大两框，让相近但不相交的两框相交  连着插入
        当框加宽=当框[0]+1.5*(当框[2]-当框[0])#实为：当框[2]
        当框加高=当框[1]+1.5*(当框[3]-当框[1])#实为：当框[3]
        另框加宽=另框[0]+1.5*(另框[2]-另框[0])#实为：另框[2]
        另框加高=另框[1]+1.5*(另框[3]-另框[1])#实为：另框[3]
        交横 = max(0,(min(当框加宽,另框加宽)-max(当框[0],另框[0])))
        交纵 = max(0,(min(当框加高,另框加高)-max(当框[1],另框[1])))
        宽率 = (当框[2]-当框[0])/(另框[2]-另框[0])
        高率 = (当框[3]-当框[1])/(另框[3]-另框[1])
        return 交横*交纵 > 0 and max(宽率,高率)<2.5 and min(宽率,高率)>0.4
    def 信度调整(self,框集):#(nb,4) 过分接近了也不好，尺寸夸张了泛滥目标与谁都有染
        关系框号集,独立框号集,似然框号集=[],[],[] # 添加符合角度规律提示 ↓
        框集 = torch.cat((框集,torch.zeros(框集.shape[0],1).to(框集.device)),dim=1)
        for i in range(框集.shape[0]):
            当框 = 框集[i]; d=当框[4]; 当框[4]=d/2; 关否=0
            for j in range(框集.shape[0]):
                if j==i: continue
                另框 = 框集[j]; l=torch.tensor(1)
                if self.相交相似(当框,另框):
                    x=另框[0]+另框[2]-当框[0]-当框[2]; y=另框[1]+另框[3]-当框[1]-当框[3]
                    # 旋号=((torch.atan2(l,x/y)*180/3.14+11.25)/22.5).long()
                    旋角=torch.atan2(l,x/y)*180/3.14#允许目标与预测相差在1内(原0.2)
                    当预=当框[6]*22.5+11.25; 另预=另框[6]*22.5+11.25
                    if 旋角>157.5 or 旋角<22.5:#注意到两极互联，有则取大极归负连小极
                        if 旋角>157.5: 旋角=旋角-180
                        if 当预>90: 当预=当预-180
                        if 另预>90: 另预=另预-180
                    服当=torch.abs(旋角-当预)<22.5; 服另=torch.abs(旋角-另预)<22.5
                    if 服当 or 服另:
                    # if 旋号==当框[6] or 旋号==另框[6]:#提当框信度,不能提前走就减
                        当框[4]=torch.min(torch.max(3*d,2*d+0.2*l),0.8*l)
                        关系框号集.append(i); 关否=1; 框集[i,-1]=1; break
            if 关否==0: 独立框号集.append(i)
        for i in 独立框号集:
            for j in 关系框号集:
                if self.相交相似(框集[i],框集[j]):
                    框集[i,4] *= 2; break
        return 框集# [x1, y1, x2, y2, confidence, class, rotation]

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        for i, image_pred in enumerate(prediction):
            # class_conf, class_pred = torch.max(image_pred[:, 4:4 + num_classes], 1, keepdim=True)
            ang_conf = torch.argmax(image_pred[:, -8:], 1, keepdim=True)
            # class_pred = class_pred[conf_mask]
            ang_conf = ang_conf[conf_mask]
            detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float(),ang_conf.float()), 1)#参照下句
            #   获得预测结果中包含的所有种类 
        # for c in unique_labels:
        #     detections_class = detections[detections[:, -1] == c]
        for c in [1]:
            detections_class = detections
        # max_detections = detections_class[keep] #此句不用注释
            max_detections = self.信度调整(max_detections)
# ------------------------(yolo.py,可视化)---------------------------------#
class YOLO(object):
    "model_path"        : '检头.pth',#自训的
    def detect_image(self, image, crop = False, count = False):
            # top_label   = np.array(results[0][:, 5], dtype = 'int32')
            aug_conf   = np.array(results[0][:, 6], dtype = 'int32')
            hit_mask   = np.array(results[0][:, -1])
            # for i, c in list(enumerate(top_label)):
            #     score           = top_conf[i]
            aug             = aug_conf[i] #连着插入
            hit             = hit_mask[i]
            # label = '{} {:.2f}'.format(predicted_class, score)
            if hit==0:label = '{:.1f} {}'.format((score*10), aug)
            else:label = '{:.1f} {} *'.format((score*10), aug)
# ------------------------(train.ipynb)---------------------------------#
    model_path      = '检头.pth'#自训的

# 训练部分
# ------------------------(正文:损失)---------------------------------#
import torch,random #填补random库来不定时输出角损
class Loss:
    def 角径1(self, gt,n): #算所有框的中心点 [nb,4]  两角径函连着插入
        中点集=torch.stack([torch.tensor([(b[0]+b[2]),(b[1]+b[3])]) for b in gt/2]);角度集=[]
        for i in range(gt.shape[0]):#[nb,4]需要指定是在第二维度得[nb,],[1,4]则不需 
            当前点 = 中点集[i]; 框距=torch.norm(中点集-当前点,dim=1)
            框距[i]=float('inf'); 最近点 = 中点集[torch.argmin(框距)]#略自
            角度集.append(torch.atan2(torch.tensor(1),(最近点[0]-当前点[0])/(最近点[1]-当前点[1]))*180/math.pi)# 弧度制转角度制
            if (i==gt.shape[0]-1)*(gt.shape[0]<n):
                for j in range(n-i-1):角度集.append(torch.tensor(0))
        return torch.stack(角度集).to(gt.device)
    def 角径(self, gt):# 实战中输入加上了批数为[b,n,4]，并需结合掩码
        角度集=[] #此句不能合并到入参中，因为这样下次算损时会结合上次的最大框数
        if gt.dim()==2:return 角径1(gt)
        b,n,_=gt.shape
        for i in range(b):
            待处=gt[i][gt[i].sum(1,False)>0]
            角度=self.角径1(待处,n)
            角度集.append(角度)
        角度集=torch.stack(角度集).to(gt.device)
        角度集=((角度集+11.25)/22.5).long()
        角度集*=(角度集!=8)#torch.clamp(,min=0,max=8)
        return 角度集 #在最末尾与距离集堆叠

    def __call__(self, preds, batch):
        # device  = preds[1].device
        loss    = torch.zeros(4, device=device) 
        # feats   = preds[2] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores, 预测角矢 = torch.cat([xi.view(feats[0].shape[0], self.no+8, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc,8), 1) 
        # pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        预测角矢 = 预测角矢.permute(0, 2, 1).contiguous()#b,8400,8
        # mask_gt                 = gt_bboxes.sum(2, keepdim=True).gt_(0)
        角径 = self.角径(gt_bboxes)
        # pred_bboxes             = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ ,目标角矢= self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels,角径, gt_bboxes, mask_gt
        )  #具体见后TaskAlignedAssigner类
        loss[3] = 0.01*self.bce(预测角矢, 目标角矢.to(dtype)).sum() / target_scores_sum
        if random.random()<0.03:print("角损:",loss[3].item())
class TaskAlignedAssigner(nn.Module):
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels,角径, gt_bboxes, mask_gt): #注意入参有添加“角径",下一函数亦然
        target_labels, target_bboxes, target_scores,目标角矢 = self.get_targets(gt_labels, 角径, gt_bboxes, target_gt_idx, fg_mask)
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx,目标角矢
    def get_targets(self, gt_labels,角径, gt_bboxes, target_gt_idx, fg_mask):
        # target_labels   = gt_labels.long().flatten()[target_gt_idx]
        目标角矢   = F.one_hot(角径.long().flatten()[target_gt_idx],8)
        return target_labels, target_bboxes, target_scores,目标角矢
```