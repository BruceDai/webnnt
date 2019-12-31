const fs = require('fs');
const path = require('path');

list = ['squeezenet1.1', 'mobilenetv2-1.0', 'resnet50v1', 'resnet50v2'];
result = "describe('CTS Real Model Test', function() {" + '\n' + '  ' + "const assert = chai.assert;" + '\n' + '  ' + "const nn = navigator.ml.getNeuralNetworkContext();";
for (let i = 0; i < list.length; i++) {
  let filePath1 = path.join(__dirname, `${list[i]}`, `${list[i]}.txt`);
  let data = fs.readFileSync(filePath1);
  data = JSON.parse(data);
  let filePath2 = path.join(__dirname, `${list[i]}`);
  for (let i = 0; i < data.length; i++) {
    let filePath = path.join(filePath2, `${data[i]}`);
    data_model = fs.readFileSync(filePath);
    data_model = data_model.toString();
    data_model = data_model.slice(131, -3);
    result += data_model;
  };
};
result += '});';
fs.writeFile('../../onnx-realmodel-test.js', result, { 'flag': 'a' }, function(err) {
  if (err) {
      throw err;
  }
});
