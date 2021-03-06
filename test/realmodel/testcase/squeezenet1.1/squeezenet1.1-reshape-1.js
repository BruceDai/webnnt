describe('CTS Real Model Test', function() {
    const assert = chai.assert;
    const nn = navigator.ml.getNeuralNetworkContext();
    it('Check result for layer-39 RESHAPE example/1 of squeezenet1.1 model', async function() {
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
      let data = {"layer": "layer-39", "Model": "squeezenet1.1", "Ops": "RESHAPE", "avg": avg, "bias": "null", "weight": "null", "input dimensions": [1,1,1,1000], "output dimensions": [1,1000], "stride": "null", "filter": "null", "padding": "null", "activation": "null", "axis": "null", "shapeLen": [2], "shapeValues": [1,1000]}
      data = JSON.stringify(data);
      document.getElementById("avg").insertAdjacentText("beforeend", data);
      document.getElementById("avg").insertAdjacentText("beforeend", ",");
      for (let i = 0; i < type2_length; ++i) {
        assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
      }
    });
  });