from .base import *

class TaskCopy(TaskBase):
    def __init__(self,cfg,mode):

        # Register Task
        super(TaskCopy,self).__init__('copy',cfg['mark'],mode)

        self.seq_width = cfg['seq_width'] or 8
        inp_dim = self.seq_width
        outp_dim = self.seq_width
        ctrl_size = cfg['ctrl_size'] or 100 
        ctrl_num_layers = cfg['ctrl_num_layers'] or 1
        N = cfg['mem_size'] or 128
        M = cfg['mem_dim'] or 20
        num_heads = cfg['num_heads'] or 1

        self.num_batches = cfg['batch'] or 48000
        self.batch_size = cfg['batch_size'] or 1

        self.net = NTM(inp_dim,outp_dim,ctrl_size,ctrl_num_layers,N,M,num_heads)
        self.optimizer = optim.RMSprop(self.net.parameters(),lr=1e-4,momentum=0.9,alpha=0.95)

        if mode == 'eval':
            path = 'log/{}/{}/{}.model'.format(self.name,self.mark,self.num_batches)
            print(path)
            self.net.load_state_dict(torch.load(path))

        # No speed improvement...@@
        # if torch.cuda.is_available():
            # print("[INFO] Use GPU...")
            # self.net.cuda()
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')

        
    def train(self):

        for idx,(X,y) in enumerate(tqdm(self._data_gen(
                    self.num_batches,self.batch_size),total=self.num_batches)):
            loss,cost = self._train_batch(X,y)
            if (idx+1) % 200 == 0:
                self.analysis(loss,cost,idx)

            if (idx+1) % 1000 == 0:
                super(TaskCopy,self).save_model(self.net,idx+1)
        print("[INFO] Training finished!")

    def analysis(self,loss,cost,idx):

        # Generate some testcase
        seq_len_t1 = 30
        seq_len_t2 = 15
        pred,te_y = self.evaluate(seq_len_t1,idx)
        super(TaskCopy,self).draw_sample(pred,te_y,idx)
        # pred2,te_y2 = self.evaluate(seq_len_t2,idx)
        # super(TaskCopy,self).draw_sample(pred2,te_y2,idx)
        
        test_cost = torch.sum(torch.abs(torch.round(pred)-te_y))/(self.batch_size*seq_len_t1)
        # Draw loss,cost curve
        super(TaskCopy,self).report(loss,cost,test_cost,idx)

    def evaluate(self,seq_len,idx):
        for te_X,te_y in self._data_gen(1,1,seq_len):
            pred = torch.Tensor(te_y.size())
            self.net.init(1)
            for i in range(seq_len+1):
                self.net(te_X[i])

            for o in range(seq_len):
                pred[o],_ = self.net(torch.zeros(1,self.seq_width))
            te_y = torch.transpose(te_y,0,2)
            pred = torch.transpose(pred,0,2)

        return pred.view(1,1,self.seq_width,-1), te_y.view(1,1,self.seq_width,-1)
        
    def _train_batch(self,X,y):
        self.optimizer.zero_grad()
        seq_len = y.size(0)
        pred = torch.Tensor(y.size())
        
        self.net.init(self.batch_size)

        for i in range(seq_len + 1):
            self.net(X[i])

        ## Retrieve the output
        for o in range(seq_len):
            # Feed dummy input
            pred[o],_ = self.net(torch.zeros(self.batch_size,self.seq_width))

        criterion = nn.BCELoss()

        loss = criterion(pred,y)
        loss.backward()
        super(TaskCopy,self).clip_grads(self.net)
        self.optimizer.step()
        cost = torch.sum(torch.abs(torch.round(pred)-y))

        return loss ,cost/(self.batch_size*seq_len)

    def _data_gen(self,num_batches,batch_size,seq_len=None):
        for batch in range(num_batches):
            if seq_len is None:
                seq_len = random.randint(3,20)
            dist = torch.distributions.binomial.Binomial(1,torch.ones(seq_len+1,batch_size,self.seq_width) * 0.5)
            data = dist.sample()
            y = data.clone()[:-1,:,:]
            data[seq_len,:,:] = 1.0 # delimiter

            yield data,y
