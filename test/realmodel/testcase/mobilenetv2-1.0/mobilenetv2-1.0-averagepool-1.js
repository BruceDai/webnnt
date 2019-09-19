describe('CTS Real Model Test', function() {
    const assert = chai.assert;
    const nn = navigator.ml.getNeuralNetworkContext();
    it('Check result for layer-47 AVERAGE_POOL_2D example/1 of mobilenetv2-1.0 model', async function() {
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
      let list = [];
      iterations_all = Number(options.iterations) + 1;
      for (let i = 0; i < iterations_all; i++) {
        let tStart = performance.now();
        await execution.startCompute();
        let computeTime = performance.now() - tStart;
        list.push(computeTime);
      };
      list.shift();
      let d = list.reduce((d, v) => {
        d.sum += v;
        return d;
      }, {
        sum: 0,
      });
      let avg = d.sum/list.length;
      let data = {"layer": "layer-47", "Model": "mobilenetv2-1.0", "Ops": "AVERAGE_POOL_2D", "avg": avg, "bias": "null", "weight": "null", "input dimensions": [1,7,7,1280], "output dimensions": [1,1,1,1280], "stride": [1], "filter": [7], "padding": [0], "activation": [0], "axis": "null", "shapeLen": "null", "shapeValues": "null"}
      data = JSON.stringify(data);
      document.getElementById("avg").insertAdjacentText("beforeend", data);
      document.getElementById("avg").insertAdjacentText("beforeend", ",");
      for (let i = 0; i < type2_length; ++i) {
        assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
      }
    });
  });