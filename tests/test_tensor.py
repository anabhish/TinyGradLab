
import numpy as np
from tinygradlab.engine.tensor import Tensor


def test_tensor_stores_data():
    tensor = Tensor([1.0, 2.0, 3.0])

    assert tensor.data.tolist() == [1.0, 2.0, 3.0]
    assert tensor.grad is None
    assert tensor.requires_grad is False


def test_tensor_can_track_gradients():
    tensor = Tensor(3.0, requires_grad=True)

    assert tensor.data == 3.0
    assert tensor.requires_grad is True
    assert tensor.grad is None


def test_tensor_addition():
    x = Tensor(3.0, requires_grad=True)
    y = Tensor(4.0, requires_grad=True)

    z = x + y

    assert z.data == 7.0
    assert z._op == "add"
    assert x in z._parents
    assert y in z._parents


def test_addition_backward():
    x = Tensor(3.0, requires_grad=True)
    y = Tensor(4.0, requires_grad=True)

    z = x + y
    z.backward()

    assert x.grad == 1.0
    assert y.grad == 1.0


def test_gradient_accumulation():
    x = Tensor(3.0, requires_grad=True)

    z = x + x
    z.backward()

    assert x.grad == 2.0


def test_tensor_multiplication():
    x = Tensor(3.0, requires_grad=True)
    y = Tensor(4.0, requires_grad=True)

    z = x * y

    assert z.data == 12.0
    assert z._op == "mul"


def test_multiplication_backward():
    x = Tensor(3.0, requires_grad=True)
    y = Tensor(4.0, requires_grad=True)

    z = x * y
    z.backward()

    assert x.grad == 4.0
    assert y.grad == 3.0


def test_addition_broadcasting_backward():
    x = Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    bias = Tensor([10.0, 20.0], requires_grad=True)

    result = x + bias
    result.grad = np.ones_like(result.data)
    result._backward()

    assert np.array_equal(
        bias.grad,
        np.array([2.0, 2.0]),
    )


def test_tensor_subtraction():
    x = Tensor([5.0, 8.0])
    y = Tensor([2.0, 3.0])

    result = x - y

    assert np.array_equal(result.data, np.array([3.0, 5.0]))


def test_subtraction_backward():
    x = Tensor(5.0, requires_grad=True)
    y = Tensor(2.0, requires_grad=True)

    result = x - y
    result.backward()

    assert np.allclose(x.grad, 1.0)
    assert np.allclose(y.grad, -1.0)


def test_subtraction_broadcasting_backward():
    x = Tensor([[5.0, 8.0], [6.0, 9.0]], requires_grad=True)
    bias = Tensor([1.0, 2.0], requires_grad=True)

    result = x - bias
    loss = result.sum()  # This test requires a sum() method on Tensor.
    loss.backward()

    assert np.allclose(x.grad, np.ones((2, 2)))
    assert np.allclose(bias.grad, np.array([-2.0, -2.0]))



def test_tensor_division():
    x = Tensor(8.0)
    y = Tensor(2.0)

    result = x / y

    assert np.allclose(result.data, 4.0)


def test_division_backward():
    x = Tensor(8.0, requires_grad=True)
    y = Tensor(2.0, requires_grad=True)

    result = x / y
    result.backward()

    assert np.allclose(x.grad, 0.5)
    assert np.allclose(y.grad, -2.0)

def test_matrix_multiplication():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    result = a @ b

    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(result.data, expected)


def test_matrix_multiplication_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    loss = (a @ b).sum()
    loss.backward()

    output_grad = np.ones((2, 2))
    expected_a_grad = output_grad @ b.data.T
    expected_b_grad = a.data.T @ output_grad

    assert np.allclose(a.grad, expected_a_grad)
    assert np.allclose(b.grad, expected_b_grad)



def test_relu_forward():
    x = Tensor([-2.0, 0.0, 3.0])

    result = x.relu()

    assert np.allclose(result.data, [0.0, 0.0, 3.0])


def test_relu_backward():
    x = Tensor([-2.0, 0.0, 3.0], requires_grad=True)

    # Use sum() to make the output a scalar.
    loss = x.relu().sum()
    loss.backward()

    assert np.allclose(x.grad, [0.0, 0.0, 1.0])


def test_tanh_forward():
    x = Tensor([-1.0, 0.0, 1.0])

    result = x.tanh()

    assert np.allclose(
        result.data,
        np.tanh(np.array([-1.0, 0.0, 1.0])),
    )


def test_tanh_backward():
    x = Tensor([0.0, 1.0], requires_grad=True)

    loss = x.tanh().sum()
    loss.backward()

    expected = 1.0 - np.tanh(np.array([0.0, 1.0])) ** 2

    assert np.allclose(x.grad, expected)



def test_sigmoid_forward():
    x = Tensor([-1000.0, 0.0, 1000.0])

    result = x.sigmoid()

    assert np.allclose(result.data, [0.0, 0.5, 1.0])
    assert np.all(np.isfinite(result.data))


def test_sigmoid_backward():
    x = Tensor([0.0, 1.0], requires_grad=True)

    loss = x.sigmoid().sum()
    loss.backward()

    expected_output = 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0])))
    expected_grad = expected_output * (1.0 - expected_output)

    assert np.allclose(x.grad, expected_grad)


def test_multiplication_gradient_matches_finite_difference():
    x = Tensor(3.0, requires_grad=True)
    y = Tensor(4.0, requires_grad=True)

    loss = (x * y).sum()
    loss.backward()

    epsilon = 1e-5

    def numerical_gradient(x_value):
        plus = Tensor(x_value + epsilon) * Tensor(4.0)
        minus = Tensor(x_value - epsilon) * Tensor(4.0)

        return (
            plus.data.item() - minus.data.item()
        ) / (2.0 * epsilon)

    numerical = numerical_gradient(3.0)

    assert np.allclose(x.grad.item(), numerical, atol=1e-6)


def test_gelu_forward():
    x = Tensor([-2.0, 0.0, 2.0])

    result = x.gelu()

    alpha = np.sqrt(2.0 / np.pi)
    beta = 0.044715
    u = alpha * (x.data + beta * x.data**3)
    expected = 0.5 * x.data * (1.0 + np.tanh(u))

    assert np.allclose(result.data, expected)


def test_gelu_gradient_matches_finite_difference():
    x = Tensor(0.7, requires_grad=True)

    loss = x.gelu().sum()
    loss.backward()

    epsilon = 1e-5

    def calculate_loss(value):
        u = np.sqrt(2.0 / np.pi) * (
            value + 0.044715 * value**3
        )
        return 0.5 * value * (1.0 + np.tanh(u))

    numerical = (
        calculate_loss(0.7 + epsilon)
        - calculate_loss(0.7 - epsilon)
    ) / (2.0 * epsilon)

    assert np.allclose(x.grad.item(), numerical, atol=1e-6)


def test_sum_axis_forward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    assert np.allclose(x.sum(axis=0).data, [4.0, 6.0])
    assert np.allclose(x.sum(axis=1).data, [3.0, 7.0])


def test_sum_axis_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    loss = x.sum(axis=0).sum()
    loss.backward()

    assert np.allclose(x.grad, np.ones((2, 2)))


def test_repeated_backward_does_not_reuse_stale_gradients():
    x = Tensor(3.0, requires_grad=True)
    loss = x * x

    loss.backward()
    first_grad = x.grad.item()

    loss.backward()
    second_grad = x.grad.item()

    assert np.allclose(first_grad, 6.0)
    assert np.allclose(second_grad, 6.0)
