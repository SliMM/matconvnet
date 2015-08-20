function insertLayer(obj, index, name, block, inputs, outputs, params)
% OBJ.INSERTLAYER(NAME, LAYER, INPUTS, OUTPUTS, PARAMS) inserts the
% specified layer in the network at the given index. INDEX must be the rank
% in topological order, NAME is a string  with the layer name, used as a
% unique indentifier. BLOCK is the object implementing the layer, which
% should be a subclass of the Layer. INPUTS, OUTPUTS are cell arrays of
% variable names, and PARAMS of parameter names.
%
%
% There are three ways in which a variable V can be attached to the input
% of a new layer L:
%
%   1. V can be disconnected from all the layers it's an input for and
%      connected to an input of L;
%   2. V can be connected to this layer in addition to the other layers
%      it's connected to;
%   3. V can "pass through" this layer: it will be connected to L and one
%      of the output variables U from this layer will connect to all the
%      inputs V was connected to.
%
%  In other words, if before L is inserted into the network:
%     - sinks(V) are the layers V is an input to; and
%     - inputs(M) are the inputs to layer M.
%  then, after L is inserted into the network we have, respecitvely:
%     1. sinks'(V) = L; or
%     2. sinks'(V) = sinks(V) + L; or
%     3. sinks'(V) = L and for all M in sinks(V), U is in inputs'(M)

f = find(strcmp(name, {obj.layers.name}), 1) ;
if ~isempty(f), error('There is already a layer with name ''%s''.', name), end
f = index ;

if nargin < 7, params = {} ; end
if ischar(inputs), inputs = {inputs} ; end
if ischar(outputs), outputs = {outputs} ; end
if ischar(params), params = {params} ; end

for i=1:numel(inputs)
  input = inputs{i};
  output = outputs{i};

  if isempty(output)
    varAction = 'disconnect';
  elseif strcmp(input, output)
    varAction = 'copy';
  else
    varAction = 'transform';
  end

  v = obj.addVar(char(input)) ;

  switch varAction
    case 'disconnect'
      % find the layers v is an input to
      layers = find(cellfun(@(x) any(x == v), {obj.layers.inputIndexes}));

      for l=layers
        obj.layers(l).inputs = ...
          obj.layers(l).inputs(~strcmp(obj.layers(l).inputs, input));

        obj.layers(l).inputIndexes = ...
          obj.layers(l).inputIndexes(obj.layers(l).inputIndexes ~= v);
      end

      obj.vars(v).fanout = 1 ;

    case 'copy'
      obj.vars(v).fanout = obj.vars(v).fanout + 1 ;

    case 'transform'
      u = obj.addVar(char(output));
      obj.vars(u).fanin = obj.vars(u).fanin + 1 ;
      obj.vars(u).fanout = obj.vars(u).fanout + obj.vars(v).fanout ;

      % find the layers v is an input to
      layers = find(cellfun(@(x) any(x == v), {obj.layers.inputIndexes}));

      for l=layers
        obj.layers(l).inputs(strcmp(obj.layers(l).inputs, input)) = {output};
        obj.layers(l).inputIndexes(obj.layers(l).inputIndexes == v) = u;
      end

      obj.vars(v).fanout = 1 ;
  end
end

for i=numel(inputs)+1:numel(outputs)
  v = obj.addVar(char(output)) ;
  obj.vars(v).fanin = obj.vars(v).fanin + 1 ;
end

for param = params
  p = obj.addParam(char(param)) ;
  obj.params(p).fanout = obj.params(p).fanout + 1 ;
end

newLayer = struct(...
  'name', {name}, ...
  'inputs', {inputs}, ...
  'outputs', {outputs}, ...
  'params', {params}, ...
  'inputIndexes', {[]}, ...
  'outputIndexes', {[]}, ...
  'paramIndexes', {[]}, ...
  'block', {block}) ;

if isfield(obj.layers, 'forwardTime')
  newLayer.forwardTime = 0;
end

obj.layers = [obj.layers(1:f-1), newLayer, obj.layers(f:end)];

block.net = obj ;

obj.rebuild() ;

end

