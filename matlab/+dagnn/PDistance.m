classdef PDistance < dagnn.Filter

  properties
    p = 2
    noRoot = false
    epsilon = 1e-6
  end
  
  methods
    function outputs = forward(obj, inputs, ~)
      outputs{1} = vl_nnpdist(...
        inputs{1}, inputs{2}, obj.p, ...
        'noroot', obj.noRoot, ...
        'epsilon', obj.epsilon) ;
    end
    
    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
      derInputs{1} = vl_nnpdist(...
        inputs{1}, inputs{2}, obj.p, derOutputs{1}, ...
        'noroot', obj.noRoot, ...
        'epsilon', obj.epsilon) ;
      derInputs{2} = [];
      derParams = {};
    end
        
    function obj = PDistance(varargin)
      obj.load(varargin) ;
    end
  end
  
end

