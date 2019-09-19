describe('CTS Real Model Test', function() {
    const assert = chai.assert;
    const nn = navigator.ml.getNeuralNetworkContext();
    it('Check result for layer-24 CONCATENATION example/5 of squeezenet1.1 model', async function() {
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
      let list = [];
      iterations_all = Number(options.iterations) + 1;
      for (let i = 0; i < iterations_all; i++) {
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
      let data = {"layer": "layer-24", "Model": "squeezenet1.1", "Ops": "CONCATENATION", "avg": avg, "bias": "null", "weight": "null", "input dimensions": [1,13,13,192], "output dimensions": [1,13,13,384], "stride": "null", "filter": "null", "padding": "null", "activation": "null", "axis": [3], "shapeLen": "null", "shapeValues": "null"}
      data = JSON.stringify(data);
      document.getElementById("avg").insertAdjacentText("beforeend", data);
      document.getElementById("avg").insertAdjacentText("beforeend", ",");
      for (let i = 0; i < type3_length; ++i) {
        assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
      }
    });
  });