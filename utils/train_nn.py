""" Template of training NN function"""

from utils.persistence import *
import time, datetime, pytz
from visualizers.metrics import Metrics
from datetime import datetime as dt

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, recall_score, precision_score


metrics = Metrics()
acc_metrics = Metrics()

def cur_time_fn():
    return dt.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M")

def train_nn (N_EPOCHS):
    
    with open(logs_path, "a+") as logs:
                cur_time = cur_time_fn()
                logs.write("\n\nEval {}, \nCurrent time {} \n"\
               .format(eval_name, cur_time))
                
    losses = [] 
    losses_val = []
    accs = []
    accs_val = []
    epoch = 1

    t_start = time.time()
    
    for i in range(N_EPOCHS):
        
        print "\nepoch", epoch
        n_batches = 0
        t0 = time.time()
        
        ## generate train batch
        data, target, _ = train_generator.next()
        target_flat = target.flatten()
        ## weights for target elements
        weights_target = np.where(target_flat<0.5, nonlesion_weight, lesion_weight).astype(np.float32)
        ## compute loss for train batch 
        ce_i, reg_i, acc_i, max_pred_i, min_pred_i = \
                train_func(data.astype(np.float32),target_flat,weights_target)   
        
        print "sum, max_pred, min_pred", max_pred_i[0]+min_pred_i[1], max_pred_i, min_pred_i
        ## sanity check of metrics
        print 'Loss sanity check: ce_i %.3f, reg_i %.3f, acc_i %.3f'%(ce_i,reg_i,acc_i)    
            
        ## approximate class balance info
        if epoch==1:
            nmb_nonles, nmb_les = np.sum(target==0), np.sum(target==1)
            class_balance_info = "\nClass Balance for 1 epoch: rate {:.2f}, nonlesions {}, lesions {}\n"\
                                    .format(nmb_les*1./nmb_nonles, nmb_nonles, nmb_les)            
            print class_balance_info
            with open(logs_path, "a+") as logs:
                logs.write(class_balance_info)

        loss_i = ce_i
        metrics["train loss"][epoch] = loss_i
#         metrics["train reg"][epoch] = reg_i
        acc_metrics["train acc"][epoch] = acc_i
        losses.append(loss_i) 
        accs.append(acc_i)
            
        ## save metrics for train and validation
        if epoch%10==0:
            
            ## validation batch
            data, target, masked_data = validation_generator.next()
            target_flat = target.flatten()
            weights_target = np.where(target_flat<0.5, nonlesion_weight, lesion_weight).astype(np.float32)
            ce_i, reg_i, acc_i, max_pred_i, min_pred_i = \
                    eval_func(data.astype(np.float32), target_flat, weights_target)
            loss_i = ce_i
            
            metrics["test loss"][epoch] = loss_i
#             metrics["test reg"][epoch] = reg_i 
            acc_metrics["test acc"][epoch] = acc_i
            losses_val.append(loss_i)
            accs_val.append(acc_i)

            print('#'*30)
            mean_train_loss = np.mean(losses[-10:])
            print('%d \t| ce\t| %.3f\t| %.3f'%(epoch, mean_train_loss, losses_val[-1]))
#             print('%d \t| auc| %.3f\t| %.3f'%(epoch, np.mean(metrics["train auc"][-10:]), auc_i))
            print('#'*30)
            
            with open(logs_path, "a+") as logs:
                cur_time = cur_time_fn()
                logs.write("""\nEpoch %d\t| Train Loss %.3f\t| Val Loss %.3f\t| Train ACC %.3f\t| Val ACC %.3f\t |Cur time: %s"""\
               %(epoch, mean_train_loss, losses_val[-1], np.mean(accs[-10:]), acc_i, cur_time))
        
        if epoch%200==0:
            ## plot and save metrics evaluation
            fig = plt.figure(figsize=[15,5])
            name = eval_name+"_metrics{}epoch.png".format(epoch)
            path_save_plot = join(path_metrics_plot,name)
            metrics.plot()
            fig.savefig(path_save_plot)
            fig = plt.figure(figsize=[15,5])
            name = eval_name+"_ACCmetrics{}epoch.png".format(epoch)
            path_save_plot = join(path_metrics_plot,name)
            acc_metrics.plot()
            fig.savefig(path_save_plot)

            ## weights snapshort
            name = eval_name + '_weights{}epoch'.format(epoch) + '.pickle'
            file_weights_path = join(path_snapshots, name)
            save(cnn.outlayer_for_loss, file_weights_path)
            
            ## visualize small predictions
            name = eval_name+'_pred_small_img{}epoch'.format(epoch)
            plot_predictions(cnn.predict,validation_generator,info=True, n_images=NMB_SAMPLES,
                              path_to_save=join(path_pred_small_img, name))
        
        ## predict full validation image
        if epoch%400==0:
            with open(logs_path, "a+") as logs:
                logs.write("""\nPredicting full image...\t| Cur time: %s"""%(cur_time_fn()))
                
            data, target, masked_data = valid_fullimg_gen.next()
            
            print cur_time_fn(), 'Predicting full image...'
            pred = predict_full_image(data, model=cnn) 
            print cur_time_fn(), 'Done. Process for plot...'
            
            with open(logs_path, "a+") as logs:
                logs.write("""\nDone.\t| Cur time: %s\t| Plotting..."""%(cur_time_fn()))

            # if None transform to list for iteration
            if masked_data is None:
                masked_data=[None]*len(data)
            
            for i, (d, s, md, p) in enumerate(zip(data, target, masked_data, pred)):
                p_les = p[1]
                p_nles = p[0]
                if md is not None:
                    img_edged, gt_edged, plot_img_gt, plot_img_pred, diff_gt_pred = \
                    process_img_for_plot(d.transpose(1,2,0), s[0], p_les, 0.3, md.transpose(1,2,0))
                else:
                    img_edged, gt_edged, plot_img_gt, plot_img_pred, diff_gt_pred = \
                    process_img_for_plot(d.transpose(1,2,0), s[0], p_les, rate=0.3)
        
                np.save(join(path_pred_full_img,'pred{}epoch{}'.format(epoch,i)), p_les)
                np.save(join(path_pred_full_img,'pred_oppos{}epoch{}'.format(epoch,i)), p_nles)
                np.save(join(path_pred_full_img,'img_edged{}epoch{}'.format(epoch,i)), img_edged)
                np.save(join(path_pred_full_img,'gt_edged{}epoch{}'.format(epoch,i)), gt_edged)
                
                name = 'pred_full_img{}epoch{}.png'.format(epoch,i)
                plot_dsp(p_les, img_edged, gt_edged, plot_img_gt, plot_img_pred, diff_gt_pred, \
                         pred_map2 = p_nles, bin_pred = p.argmax(axis=0), path_to_save=join(path_pred_full_img,name))
                print i, 'saved and ploted'                
                if i > NMB_SAMPLES: break
            
            with open(logs_path, "a+") as logs:
                logs.write("""\nDone.\t| Cur time: %s"""%(cur_time_fn()))
            
            np.save(join(path_snapshots,'losses.npy'), losses)
            np.save(join(path_snapshots, 'losses_val.npy'), losses_val)
            np.save(join(path_snapshots,'acc.npy'), accs)
            np.save(join(path_snapshots, 'acc_val.npy'), accs_val)
            
        epoch+=1
        print 'time for epoch: %.2f mins'%((time.time() - t0)/60.)
    print 'Overall time: %.2f mins'%((time.time() - t_start)/60.)