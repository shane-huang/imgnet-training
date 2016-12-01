--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

nClasses = opt.nClasses

paths.dofile('util.lua')
paths.dofile('model.lua')
opt.imageSize = model.imageSize or opt.imageSize
opt.imageCrop = model.imageCrop or opt.imageCrop

print(opt)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

-- should put before train.lua - getParameters() 
-- it works only put before that line, still investigating why
if opt.netType == "googlenet" then
      if torch.type(model) == 'nn.DataParallelTable' then
         optmodel = model:get(1)
      else
         optmodel = model
      end

      local optnet = require 'optnet'
      -- local imsize = opt.dataset == 'imagenet' and 224 or 32
      local imsize = opt.imageCrop
      local sampleInput = torch.zeros(4,3,imsize,imsize):cuda()
      print('Start optimizing memory...')
      optnet.optimizeMemory(optmodel,sampleInput,{inplace=false,mode='training'})
      print('Done optimizing memory.')
else
      print('Memory optimization do not support ' .. opt.netType .. '. Omitted.')
end

paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

epoch = opt.epochNumber

for i=1,opt.nEpochs do
   train()
   test()
   epoch = epoch + 1
end
