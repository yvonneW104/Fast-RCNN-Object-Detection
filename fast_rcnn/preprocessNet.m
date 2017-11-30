function net = preprocessNet(net)
% Load the network and put it in test mode.
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ;

% Mark class and bounding box predictions as `precious` so they are
% not optimized away during evaluation.
net.vars(net.getVarIndex('cls_prob')).precious = 1 ;
net.vars(net.getVarIndex('bbox_pred')).precious = 1 ;
end