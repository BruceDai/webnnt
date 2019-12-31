describe('CTS Real Model Test', function() {
    const assert = chai.assert;
    const nn = navigator.ml.getNeuralNetworkContext();
    it('Check result for layer-91 FULLY_CONNECTED example/1 of resnet50v2 model', async function() {
      let model = await nn.createModel(options);
      let operandIndex = 0;
      let op1_value;
      let output_expect;
      await fetch('./realmodel/testcase/res/resnet50v2/971').then((res) => {
        return res.json();
      }).then((text) => {
        let file_data = new Float32Array(text.length);
        for (let j in text) {
          let b = parseFloat(text[j]);
          file_data[j] = b;
        }
        op1_value = file_data;
      });
      await fetch('./realmodel/testcase/res/resnet50v2/261').then((res) => {
        return res.json();
      }).then((text) => {
        let file_data = new Float32Array(text.length);
        for (let j in text) {
          let b = parseFloat(text[j]);
          file_data[j] = b;
        }
        output_expect = file_data;
      });
      let type4 = {type: nn.INT32};
      let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1000,2048]};
      let type1_length = product(type1.dimensions);
      let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1000]};
      let type2_length = product(type2.dimensions);
      let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1,1000]};
      let type3_length = product(type3.dimensions);
      let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1,2048]};
      let type0_length = product(type0.dimensions);

      let op1 = operandIndex++;
      model.addOperand(type0);
      let op2 = operandIndex++;
      model.addOperand(type1);
      let b0 = operandIndex++;
      model.addOperand(type2);
      let op3 = operandIndex++;
      model.addOperand(type3);
      let act = operandIndex++;
      model.addOperand(type4);
      let op2value;
      await fetch('./realmodel/testcase/res/resnet50v2/259').then((res) => {
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
      await fetch('./realmodel/testcase/res/resnet50v2/260').then((res) => {
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
      model.setOperandValue(b0, new Float32Array(op3value));
      model.setOperandValue(act, new Int32Array([0]));
      model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

      model.identifyInputsAndOutputs([op1], [op3]);
      await model.finish();

      let compilation = await model.createCompilation();
      compilation.setPreference(getPreferenceCode(options.prefer));
      await compilation.finish();

      let execution = await compilation.createExecution();

      let op1_input = new Float32Array(op1_value);
      execution.setInput(0, op1_input);

      let op3_output = new Float32Array(type3_length);
      execution.setOutput(0, op3_output);
      let list = [];
      iterations_all = Number(options.iterations) + 1;
      for (let i = 0; i < iterations_all ; i++) {
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
      let data = {"layer": "layer-91", "Model": "resnet50v2", "Ops": "FULLY_CONNECTED", "avg": avg, "bias": "null", "weight": "null", "input dimensions": [1,2048], "output dimensions": [1,1000], "stride": "null", "filter": "null", "padding": "null", "activation": [0], "axis": "null", "shapeLen": "null", "shapeValues": "null"}
      data = JSON.stringify(data);
      document.getElementById("avg").insertAdjacentText("beforeend", data);
      document.getElementById("avg").insertAdjacentText("beforeend", ",");
      for (let i = 0; i < type2_length; ++i) {
        assert.isTrue(almostEqualCTS(op3_output[i], output_expect[i]));
      }
    });
  });