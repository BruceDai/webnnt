describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
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
    let list = [];
    iterations = Number(options.iterations) + 1;
    for (let i = 0; i < iterations; i++) {
      let tStart = performance.now();
      await execution.startCompute();
      let computeTime = performance.now() - tStart;
      list.push(computeTime);
    };
    let sum = 0;
    list.shift();
    let d = list.reduce((d, v) => {
      d.sum += v;
      return d;
    }, {
      sum: 0,
    });
    let avg = d.sum/list.length;
    let data = {"layer": "layer-2", "Model": "mobilenetv2-1.0", "Ops": "CONV_2D", "avg": avg, "bias": [32], "weight": [32,1,1,32], "input dimensions": [1,112,112,32], "output dimensions": [1,112,112,32], "stride": [1], "filter": "null", "padding": [0], "activation": [1], "axis": "null", "shapeLen": "null", "shapeValues": "null"}
    data = JSON.stringify(data);
    document.getElementById("avg").insertAdjacentText("beforeend", data);
    document.getElementById("avg").insertAdjacentText("beforeend", ",");
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});