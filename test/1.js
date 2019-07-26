describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  it.only('Check result for layer-1 CONV_2D example/1 of squeezenet1.1 model', async function() {
//    this.timeout(220000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/0').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/62').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,224,224,3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,111,111,64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64,3,3,3]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/1').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/2').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([2]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-2 MAX_POOL_2D example/1 of squeezenet1.1 model', async function() {
    this.timeout(220000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let i0_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/62').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      i0_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/72').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,111,111,64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,64]};
    let type2_length = product(type2.dimensions);
    let i0 = operandIndex++;
    model.addOperand(type0);
    let stride = operandIndex++;
    model.addOperand(type1);
    let filter = operandIndex++;
    model.addOperand(type1);
    let padding = operandIndex++;
    model.addOperand(type1);
    let activation = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(stride, new Int32Array([2]));
    model.setOperandValue(filter, new Int32Array([3]));
    model.setOperandValue(padding, new Int32Array([0]));
    model.setOperandValue(activation, new Int32Array([0]));
    model.addOperation(nn.MAX_POOL_2D, [i0, padding, padding, padding, padding, stride, stride, filter, filter, activation], [output]);
    model.identifyInputsAndOutputs([i0], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let i0_input = new Float32Array(i0_value);
    execution.setInput(0, i0_input);
    let output_output = new Float32Array(type2_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-3 CONV_2D example/2 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/72').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/80').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,16]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [16]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [16,1,1,64]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/3').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/4').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-4 CONV_2D example/3 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/80').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/88').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,16]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64,1,1,16]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/5').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/6').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-5 CONV_2D example/4 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/80').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/96').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,16]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64,3,3,16]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/7').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/8').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-6 CONCATENATION example/1 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input1_value;
    let input2_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/88').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        file_data[j] = parseFloat(text[j]);
      }
      input1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/96').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      input2_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/98').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type2 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,64]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,64]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,128]};
    let type3_length = product(type3.dimensions);
    let input1 = operandIndex++;
    model.addOperand(type0);
    let input2 = operandIndex++;
    model.addOperand(type1);
    let axis0 = operandIndex++;
    model.addOperand(type2);
    let output = operandIndex++;
    model.addOperand(type3);
    let input2_input = new Float32Array(input2_value);
    model.setOperandValue(input2, input2_input);
    model.setOperandValue(axis0, new Int32Array([3]));
    model.addOperation(nn.CONCATENATION, [input1, input2, axis0], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let input1_input = new Float32Array(input1_value);
    execution.setInput(0, input1_input);
    let output_output = new Float32Array(type3_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-7 CONV_2D example/5 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/98').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/106').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,128]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,16]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [16]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [16,1,1,128]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/9').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/10').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-8 CONV_2D example/6 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/106').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/114').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,16]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64,1,1,16]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/11').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/12').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-9 CONV_2D example/7 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/106').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/122').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,16]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64,3,3,16]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/13').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/14').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-10 CONCATENATION example/2 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input1_value;
    let input2_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/114').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        file_data[j] = parseFloat(text[j]);
      }
      input1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/122').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      input2_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/124').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type2 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,64]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,64]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,128]};
    let type3_length = product(type3.dimensions);
    let input1 = operandIndex++;
    model.addOperand(type0);
    let input2 = operandIndex++;
    model.addOperand(type1);
    let axis0 = operandIndex++;
    model.addOperand(type2);
    let output = operandIndex++;
    model.addOperand(type3);
    let input2_input = new Float32Array(input2_value);
    model.setOperandValue(input2, input2_input);
    model.setOperandValue(axis0, new Int32Array([3]));
    model.addOperation(nn.CONCATENATION, [input1, input2, axis0], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let input1_input = new Float32Array(input1_value);
    execution.setInput(0, input1_input);
    let output_output = new Float32Array(type3_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-11 MAX_POOL_2D example/2 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let i0_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/124').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      i0_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/134').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,55,55,128]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,128]};
    let type2_length = product(type2.dimensions);
    let i0 = operandIndex++;
    model.addOperand(type0);
    let stride = operandIndex++;
    model.addOperand(type1);
    let filter = operandIndex++;
    model.addOperand(type1);
    let padding = operandIndex++;
    model.addOperand(type1);
    let activation = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(stride, new Int32Array([2]));
    model.setOperandValue(filter, new Int32Array([3]));
    model.setOperandValue(padding, new Int32Array([0]));
    model.setOperandValue(activation, new Int32Array([0]));
    model.addOperation(nn.MAX_POOL_2D, [i0, padding, padding, padding, padding, stride, stride, filter, filter, activation], [output]);
    model.identifyInputsAndOutputs([i0], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let i0_input = new Float32Array(i0_value);
    execution.setInput(0, i0_input);
    let output_output = new Float32Array(type2_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-12 CONV_2D example/8 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/134').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/142').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,128]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,32]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [32]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [32,1,1,128]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/15').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/16').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-13 CONV_2D example/9 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/142').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/150').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,128]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [128]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [128,1,1,32]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/17').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/18').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-14 CONV_2D example/10 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/142').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/158').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,128]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [128]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [128,3,3,32]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/19').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/20').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-15 CONCATENATION example/3 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input1_value;
    let input2_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/150').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        file_data[j] = parseFloat(text[j]);
      }
      input1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/158').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      input2_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/160').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type2 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,128]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,128]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,256]};
    let type3_length = product(type3.dimensions);
    let input1 = operandIndex++;
    model.addOperand(type0);
    let input2 = operandIndex++;
    model.addOperand(type1);
    let axis0 = operandIndex++;
    model.addOperand(type2);
    let output = operandIndex++;
    model.addOperand(type3);
    let input2_input = new Float32Array(input2_value);
    model.setOperandValue(input2, input2_input);
    model.setOperandValue(axis0, new Int32Array([3]));
    model.addOperation(nn.CONCATENATION, [input1, input2, axis0], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let input1_input = new Float32Array(input1_value);
    execution.setInput(0, input1_input);
    let output_output = new Float32Array(type3_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-16 CONV_2D example/11 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/160').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/168').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,256]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,32]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [32]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [32,1,1,256]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/21').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/22').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-17 CONV_2D example/12 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/168').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/176').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,128]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [128]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [128,1,1,32]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/23').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/24').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-18 CONV_2D example/13 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/168').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/184').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,128]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [128]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [128,3,3,32]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/25').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/26').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-19 CONCATENATION example/4 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input1_value;
    let input2_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/176').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        file_data[j] = parseFloat(text[j]);
      }
      input1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/184').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      input2_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/186').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type2 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,128]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,128]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,256]};
    let type3_length = product(type3.dimensions);
    let input1 = operandIndex++;
    model.addOperand(type0);
    let input2 = operandIndex++;
    model.addOperand(type1);
    let axis0 = operandIndex++;
    model.addOperand(type2);
    let output = operandIndex++;
    model.addOperand(type3);
    let input2_input = new Float32Array(input2_value);
    model.setOperandValue(input2, input2_input);
    model.setOperandValue(axis0, new Int32Array([3]));
    model.addOperation(nn.CONCATENATION, [input1, input2, axis0], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let input1_input = new Float32Array(input1_value);
    execution.setInput(0, input1_input);
    let output_output = new Float32Array(type3_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-20 MAX_POOL_2D example/3 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let i0_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/186').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      i0_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/196').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,27,27,256]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,256]};
    let type2_length = product(type2.dimensions);
    let i0 = operandIndex++;
    model.addOperand(type0);
    let stride = operandIndex++;
    model.addOperand(type1);
    let filter = operandIndex++;
    model.addOperand(type1);
    let padding = operandIndex++;
    model.addOperand(type1);
    let activation = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(stride, new Int32Array([2]));
    model.setOperandValue(filter, new Int32Array([3]));
    model.setOperandValue(padding, new Int32Array([0]));
    model.setOperandValue(activation, new Int32Array([0]));
    model.addOperation(nn.MAX_POOL_2D, [i0, padding, padding, padding, padding, stride, stride, filter, filter, activation], [output]);
    model.identifyInputsAndOutputs([i0], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let i0_input = new Float32Array(i0_value);
    execution.setInput(0, i0_input);
    let output_output = new Float32Array(type2_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-21 CONV_2D example/14 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/196').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/204').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,256]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,48]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [48]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [48,1,1,256]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/27').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/28').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-22 CONV_2D example/15 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/204').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/212').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,48]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,192]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [192]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [192,1,1,48]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/29').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/30').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-23 CONV_2D example/16 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/204').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/220').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,48]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,192]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [192]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [192,3,3,48]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/31').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/32').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-24 CONCATENATION example/5 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input1_value;
    let input2_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/212').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        file_data[j] = parseFloat(text[j]);
      }
      input1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/220').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      input2_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/222').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type2 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,192]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,192]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,384]};
    let type3_length = product(type3.dimensions);
    let input1 = operandIndex++;
    model.addOperand(type0);
    let input2 = operandIndex++;
    model.addOperand(type1);
    let axis0 = operandIndex++;
    model.addOperand(type2);
    let output = operandIndex++;
    model.addOperand(type3);
    let input2_input = new Float32Array(input2_value);
    model.setOperandValue(input2, input2_input);
    model.setOperandValue(axis0, new Int32Array([3]));
    model.addOperation(nn.CONCATENATION, [input1, input2, axis0], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let input1_input = new Float32Array(input1_value);
    execution.setInput(0, input1_input);
    let output_output = new Float32Array(type3_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-25 CONV_2D example/17 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/222').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/230').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,384]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,48]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [48]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [48,1,1,384]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/33').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/34').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-26 CONV_2D example/18 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/230').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/238').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,48]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,192]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [192]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [192,1,1,48]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/35').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/36').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-27 CONV_2D example/19 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/230').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/246').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,48]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,192]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [192]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [192,3,3,48]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/37').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/38').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-28 CONCATENATION example/6 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input1_value;
    let input2_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/238').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        file_data[j] = parseFloat(text[j]);
      }
      input1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/246').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      input2_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/248').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type2 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,192]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,192]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,384]};
    let type3_length = product(type3.dimensions);
    let input1 = operandIndex++;
    model.addOperand(type0);
    let input2 = operandIndex++;
    model.addOperand(type1);
    let axis0 = operandIndex++;
    model.addOperand(type2);
    let output = operandIndex++;
    model.addOperand(type3);
    let input2_input = new Float32Array(input2_value);
    model.setOperandValue(input2, input2_input);
    model.setOperandValue(axis0, new Int32Array([3]));
    model.addOperation(nn.CONCATENATION, [input1, input2, axis0], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let input1_input = new Float32Array(input1_value);
    execution.setInput(0, input1_input);
    let output_output = new Float32Array(type3_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-29 CONV_2D example/20 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/248').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/256').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,384]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64,1,1,384]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/39').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/40').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-30 CONV_2D example/21 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/256').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/264').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,256]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [256]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [256,1,1,64]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/41').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/42').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-31 CONV_2D example/22 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/256').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/272').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,256]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [256]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [256,3,3,64]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/43').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/44').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-32 CONCATENATION example/7 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input1_value;
    let input2_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/264').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        file_data[j] = parseFloat(text[j]);
      }
      input1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/272').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      input2_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/274').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type2 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,256]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,256]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,512]};
    let type3_length = product(type3.dimensions);
    let input1 = operandIndex++;
    model.addOperand(type0);
    let input2 = operandIndex++;
    model.addOperand(type1);
    let axis0 = operandIndex++;
    model.addOperand(type2);
    let output = operandIndex++;
    model.addOperand(type3);
    let input2_input = new Float32Array(input2_value);
    model.setOperandValue(input2, input2_input);
    model.setOperandValue(axis0, new Int32Array([3]));
    model.addOperation(nn.CONCATENATION, [input1, input2, axis0], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let input1_input = new Float32Array(input1_value);
    execution.setInput(0, input1_input);
    let output_output = new Float32Array(type3_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-33 CONV_2D example/23 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/274').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/282').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,512]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64,1,1,512]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/45').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/46').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-34 CONV_2D example/24 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/282').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/290').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,256]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [256]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [256,1,1,64]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/47').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/48').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-35 CONV_2D example/25 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/282').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/298').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,256]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [256]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [256,3,3,64]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/49').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/50').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-36 CONCATENATION example/8 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input1_value;
    let input2_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/290').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        file_data[j] = parseFloat(text[j]);
      }
      input1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/298').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      input2_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/300').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type2 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,256]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,256]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,512]};
    let type3_length = product(type3.dimensions);
    let input1 = operandIndex++;
    model.addOperand(type0);
    let input2 = operandIndex++;
    model.addOperand(type1);
    let axis0 = operandIndex++;
    model.addOperand(type2);
    let output = operandIndex++;
    model.addOperand(type3);
    let input2_input = new Float32Array(input2_value);
    model.setOperandValue(input2, input2_input);
    model.setOperandValue(axis0, new Int32Array([3]));
    model.addOperation(nn.CONCATENATION, [input1, input2, axis0], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let input1_input = new Float32Array(input1_value);
    execution.setInput(0, input1_input);
    let output_output = new Float32Array(type3_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-37 CONV_2D example/26 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/300').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/308').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,512]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,1000]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1000]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1000,1,1,512]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/51').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/squeezenet1.1/52').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-38 AVERAGE_POOL_2D example/1 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let i0_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/308').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      i0_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/318').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,13,13,1000]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1,1,1,1000]};
    let type2_length = product(type2.dimensions);
    let i0 = operandIndex++;
    model.addOperand(type0);
    let stride = operandIndex++;
    model.addOperand(type1);
    let filter = operandIndex++;
    model.addOperand(type1);
    let padding = operandIndex++;
    model.addOperand(type1);
    let activation = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(stride, new Int32Array([13]));
    model.setOperandValue(filter, new Int32Array([13]));
    model.setOperandValue(padding, new Int32Array([0]));
    model.setOperandValue(activation, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [i0, padding, padding, padding, padding, stride, stride, filter, filter, activation], [output]);
    model.identifyInputsAndOutputs([i0], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let i0_input = new Float32Array(i0_value);
    execution.setInput(0, i0_input);
    let output_output = new Float32Array(type2_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-39 RESHAPE example/1 of squeezenet1.1 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op3_expect;
    await fetch('./realmodel/testcase/res/squeezenet1.1/318').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/squeezenet1.1/54').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,1,1,1000]};
    let type0_length = product(type0.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1,1000]};
    let type2_length = product(type2.dimensions);
    let type1 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type1_length = product(type1.dimensions);
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(op2, new Int32Array([1,1000]));
    model.addOperation(nn.RESHAPE, [op1, op2], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op3_output = new Float32Array(type2_length);
    execution.setOutput(0, op3_output);
    await execution.startCompute();
    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('Check result for layer-1 CONV_2D example/1 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/0').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/277').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,224,224,3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,112,112,32]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [32]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [32,3,3,3]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/1').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/269').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([2]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-2 CONV_2D example/2 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/277').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/286').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,112,112,32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,112,112,32]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [32]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [32,1,1,32]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/6').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/278').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-3 CONV_2D example/3 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/296').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/305').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,112,112,32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,112,112,16]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [16]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [16,1,1,32]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/16').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/297').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-4 CONV_2D example/4 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/305').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/314').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,112,112,16]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,112,112,96]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [96]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [96,1,1,16]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/21').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/306').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-5 CONV_2D example/5 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/324').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/333').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,56,56,96]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,56,56,24]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [24]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [24,1,1,96]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/31').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/325').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-6 CONV_2D example/6 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/333').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/342').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,56,56,24]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,56,56,144]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [144]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [144,1,1,24]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/36').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/334').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-7 CONV_2D example/7 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/352').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/361').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,56,56,144]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,56,56,24]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [24]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [24,1,1,144]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/46').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/353').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-8 CONV_2D example/8 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/363').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/372').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,56,56,24]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,56,56,144]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [144]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [144,1,1,24]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/51').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/364').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-9 CONV_2D example/9 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/382').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/391').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,144]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,32]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [32]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [32,1,1,144]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/61').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/383').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-10 CONV_2D example/10 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/391').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/400').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,192]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [192]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [192,1,1,32]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/66').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/392').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-11 CONV_2D example/11 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/410').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/419').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,192]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,32]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [32]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [32,1,1,192]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/76').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/411').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-12 CONV_2D example/12 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/421').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/430').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,192]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [192]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [192,1,1,32]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/81').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/422').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-13 CONV_2D example/13 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/440').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/449').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,192]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,32]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [32]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [32,1,1,192]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/91').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/441').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-14 CONV_2D example/14 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/451').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/460').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,192]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [192]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [192,1,1,32]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/96').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/452').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-15 CONV_2D example/15 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/470').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/479').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,192]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64,1,1,192]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/106').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/471').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-16 CONV_2D example/16 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/479').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/488').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,384]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [384]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [384,1,1,64]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/111').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/480').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-17 CONV_2D example/17 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/498').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/507').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,384]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64,1,1,384]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/121').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/499').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-18 CONV_2D example/18 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/509').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/518').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,384]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [384]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [384,1,1,64]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/126').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/510').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-19 CONV_2D example/19 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/528').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/537').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,384]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64,1,1,384]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/136').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/529').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-20 CONV_2D example/20 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/539').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/548').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,384]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [384]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [384,1,1,64]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/141').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/540').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-21 CONV_2D example/21 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/558').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/567').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,384]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64,1,1,384]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/151').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/559').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-22 CONV_2D example/22 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/569').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/578').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,28,28,384]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [384]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [384,1,1,64]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/156').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/570').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-23 CONV_2D example/23 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/588').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/597').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,384]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,96]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [96]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [96,1,1,384]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/166').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/589').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-24 CONV_2D example/24 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/597').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/606').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,96]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,576]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [576]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [576,1,1,96]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/171').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/598').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-25 CONV_2D example/25 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/616').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/625').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,576]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,96]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [96]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [96,1,1,576]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/181').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/617').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-26 CONV_2D example/26 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/627').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/636').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,96]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,576]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [576]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [576,1,1,96]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/186').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/628').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-27 CONV_2D example/27 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/646').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/655').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,576]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,96]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [96]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [96,1,1,576]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/196').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/647').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-28 CONV_2D example/28 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/657').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/666').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,96]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,14,14,576]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [576]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [576,1,1,96]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/201').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/658').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-29 CONV_2D example/29 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/676').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/685').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,576]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,160]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [160]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [160,1,1,576]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/211').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/677').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-30 CONV_2D example/30 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/685').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/694').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,160]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,960]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [960]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [960,1,1,160]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/216').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/686').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-31 CONV_2D example/31 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/704').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/713').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,960]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,160]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [160]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [160,1,1,960]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/226').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/705').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-32 CONV_2D example/32 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/715').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/724').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,160]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,960]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [960]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [960,1,1,160]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/231').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/716').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-33 CONV_2D example/33 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/734').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/743').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,960]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,160]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [160]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [160,1,1,960]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/241').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/735').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-34 CONV_2D example/34 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/745').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/754').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,160]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,960]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [960]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [960,1,1,160]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/246').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/746').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-35 CONV_2D example/35 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/764').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/773').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,960]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,320]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [320]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [320,1,1,960]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/256').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/765').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-36 CONV_2D example/36 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/773').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/782').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,320]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,1280]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1280]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1280,1,1,320]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/261').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/774').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-37 AVERAGE_POOL_2D example/1 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let i0_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/782').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      i0_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/792').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,7,7,1280]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1,1,1,1280]};
    let type2_length = product(type2.dimensions);
    let i0 = operandIndex++;
    model.addOperand(type0);
    let stride = operandIndex++;
    model.addOperand(type1);
    let filter = operandIndex++;
    model.addOperand(type1);
    let padding = operandIndex++;
    model.addOperand(type1);
    let activation = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(stride, new Int32Array([1]));
    model.setOperandValue(filter, new Int32Array([7]));
    model.setOperandValue(padding, new Int32Array([0]));
    model.setOperandValue(activation, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [i0, padding, padding, padding, padding, stride, stride, filter, filter, activation], [output]);
    model.identifyInputsAndOutputs([i0], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let i0_input = new Float32Array(i0_value);
    execution.setInput(0, i0_input);
    let output_output = new Float32Array(type2_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('Check result for layer-38 CONV_2D example/37 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/792').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/801').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,1,1,1280]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1,1,1,1000]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1000]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1000,1,1,1280]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/266').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/793').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('Check result for layer-39 RESHAPE example/1 of mobilenetv2-1.0 model', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op3_expect;
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/801').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/mobilenetv2-1.0/268').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,1,1,1000]};
    let type0_length = product(type0.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1,1000]};
    let type2_length = product(type2.dimensions);
    let type1 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type1_length = product(type1.dimensions);
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(op2, new Int32Array([1,1000]));
    model.addOperation(nn.RESHAPE, [op1, op2], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op3_output = new Float32Array(type2_length);
    execution.setOutput(0, op3_output);
    await execution.startCompute();
    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});


