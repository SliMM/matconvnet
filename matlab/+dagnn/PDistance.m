classdef PDistance < dagnn.Filter

  properties
    p = 2
    noRoot = false
    epsilon = 1e-6
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end
  
  methods
    function outputs = forward(obj, inputs, ~)
      outputs{1} = vl_nnpdist(...
        inputs{1}, inputs{2}, obj.p, ...
        'noroot', obj.noRoot, ...
        'epsilon', obj.epsilon) ;
      
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + sum(gather(outputs{1}), 4)) / m ;
      obj.numAveraged = m ;
    end
    
    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
      derInputs{1} = vl_nnpdist(...
        inputs{1}, inputs{2}, obj.p, derOutputs{1}, ...
        'noroot', obj.noRoot, ...
        'epsilon', obj.epsilon) ;
      derInputs{2} = [];
      derParams = {};
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end
        
    function obj = PDistance(varargin)
      obj.load(varargin) ;
    end
  end
  
end

