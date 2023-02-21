"""
Created on Fri Mar 5 2021

@author: Kuan-Lin Chen
"""
import torch
import scipy.io
from datetime import datetime
from utils import get_device_name

def testRegressor(net,testset,criterion,device,model_folder,n_target,n_interference,eval_batch_size):
    print('{} {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),get_device_name(device)))
    torch.manual_seed(0)
    net = net.to(device)
    testloader = torch.utils.data.DataLoader(testset,batch_size=eval_batch_size,shuffle=False,num_workers=1,pin_memory=False,drop_last=False)
    net.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader,1):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += torch.sum(loss).item()
            total += loss.numel()

    avg_test_loss = test_loss/total

    print('[Test] loss: %.4f'%(avg_test_loss))

    test_result = {'avg_test_loss':avg_test_loss}
    test_result_path = '{}/test_nt={}_ni={}.mat'.format(model_folder,n_target,n_interference)
    scipy.io.savemat(test_result_path,test_result)
    print('{} test results saved at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),test_result_path))
    return avg_test_loss
