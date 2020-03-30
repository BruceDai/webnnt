describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Softmax v1_2 example-1', async function() {
    // For 'Softmax v1_2' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0, 17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0, 17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0, 17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0];
    let op2_expect = [0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 5]};
    let type0_length = product(type0.dimensions);
    let type2 = {type: nn.FLOAT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let param = operandIndex++;
    model.addOperand(type2);
    let op2 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(param, new Float32Array([1.0]));
    model.addOperation(nn.SOFTMAX, [op1, param], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Float32Array(type0_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax v1_2 example-2', async function() {
    // For 'Softmax v1_2' example: examples_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0, 17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0, 17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0, 17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0];
    let op2_expect = [0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 5]};
    let type0_length = product(type0.dimensions);
    let type2 = {type: nn.FLOAT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let param = operandIndex++;
    model.addOperand(type2);
    let op2 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(param, new Float32Array([1.0]));
    model.addOperation(nn.SOFTMAX, [op1, param], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Float32Array(type0_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax v1_2 example-3', async function() {
    // For 'Softmax v1_2' example: examples_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0, 17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0, 17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0, 17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0];
    let op2_expect = [0.6439142227172852, 0.23688280582427979, 0.08714431524276733, 0.03205860033631325, 7.246299560392799e-08, 0.6439142227172852, 0.23688280582427979, 0.08714431524276733, 0.03205860033631325, 7.246299560392799e-08, 0.6439142227172852, 0.23688280582427979, 0.08714431524276733, 0.03205860033631325, 7.246299560392799e-08, 0.6439142227172852, 0.23688280582427979, 0.08714431524276733, 0.03205860033631325, 7.246299560392799e-08, 0.6439142227172852, 0.23688280582427979, 0.08714431524276733, 0.03205860033631325, 7.246299560392799e-08, 0.6439142227172852, 0.23688280582427979, 0.08714431524276733, 0.03205860033631325, 7.246299560392799e-08, 0.6439142227172852, 0.23688280582427979, 0.08714431524276733, 0.03205860033631325, 7.246299560392799e-08, 0.6439142227172852, 0.23688280582427979, 0.08714431524276733, 0.03205860033631325, 7.246299560392799e-08];

    let type14 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 5]};
    let type14_length = product(type14.dimensions);
    let type15 = {type: nn.FLOAT32};

    let op1 = operandIndex++;
    model.addOperand(type14);
    let param = operandIndex++;
    model.addOperand(type15);
    let op2 = operandIndex++;
    model.addOperand(type14);

    model.setOperandValue(param, new Float32Array([1.0]));
    model.addOperation(nn.SOFTMAX, [op1, param], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Float32Array(type14_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type14_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax with 2D input tensor example', async function() {
    let model = await nn.createModel(options);
    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand({type: nn.FLOAT32});
    model.setOperandValue(1, new Float32Array([1.0]));
    model.addOperand(float32TensorType);
    model.addOperation(nn.SOFTMAX, [0, 1], [2]);

    model.identifyInputsAndOutputs([0], [2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([1.0, 1.0, 1.0, 1.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);

    await execution.startCompute();
    let expectedData = new Float32Array(tensorLength);
    expectedData.set([0.5, 0.5, 0.5, 0.5]);

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });
})
